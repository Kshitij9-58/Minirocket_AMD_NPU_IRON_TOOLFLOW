//===- dotprod_runner_batch.cpp --------------------------------------===//
// Persistent, batched runner for the dot-product kernel.
//
// Unlike dotprod_runner.cpp (one process per dispatch), this opens the
// device, registers the xclbin, and allocates buffers ONCE, then loops
// internally over every (sample, class) combination, reusing the same
// buffers and hardware context. Eliminates 191 of 192 process-spawn and
// xclbin-registration costs versus the per-dispatch version.
//
// Input:  features_all.txt  (num_samples rows x FEATURE_COUNT floats)
//         weights_all.txt   (num_classes rows x FEATURE_COUNT floats)
// Output: scores_all.txt    (num_samples x num_classes floats, row-major,
//                            i.e. sample i's scores are on line i,
//                            space-separated, one score per class)
//
// STATUS: untested, not yet run against hardware.
//===----------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using INPUT_TYPE = uint16_t;   // bfloat16 bits
using OUTPUT_TYPE = float;

const int FEATURE_COUNT = 840;

uint16_t float_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    return static_cast<uint16_t>(i >> 16);
}

std::vector<uint32_t> load_instr_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open instruction file: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("Invalid binary size.");
    std::vector<uint32_t> buffer(size / 4);
    if (!file.read((char*)buffer.data(), size)) throw std::runtime_error("Read failed.");
    return buffer;
}

// Reads a whitespace-separated matrix, one row per line, into a vector of
// row vectors. Every row must have exactly FEATURE_COUNT values.
std::vector<std::vector<float>> load_matrix(const std::string &filename) {
    std::ifstream file(filename);
    if (!file) throw std::runtime_error("Could not open matrix file: " + filename);
    std::vector<std::vector<float>> rows;
    std::string line;
    while (std::getline(file, line)) {
        if (line.find_first_not_of(" \t\r\n") == std::string::npos) continue; // skip blank lines
        std::istringstream iss(line);
        std::vector<float> row;
        row.reserve(FEATURE_COUNT);
        float v;
        while (iss >> v) row.push_back(v);
        if ((int)row.size() != FEATURE_COUNT) {
            throw std::runtime_error(filename + ": row has " + std::to_string(row.size()) +
                                      " values, expected " + std::to_string(FEATURE_COUNT));
        }
        rows.push_back(std::move(row));
    }
    return rows;
}

int main(int argc, char *argv[]) {
    std::string xclbin_path = "dotproduct.xclbin";
    std::string insts_path = "dotproduct_insts.txt";
    std::string kernel_name = "MLIR_AIE";
    std::string features_file = "features_all.txt";
    std::string weights_file = "weights_all.txt";
    std::string output_file = "scores_all.txt";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) xclbin_path = argv[++i];
        else if (arg == "-i" && i + 1 < argc) insts_path = argv[++i];
        else if (arg == "-k" && i + 1 < argc) kernel_name = argv[++i];
        else if (arg == "-F" && i + 1 < argc) features_file = argv[++i];
        else if (arg == "-W" && i + 1 < argc) weights_file = argv[++i];
        else if (arg == "-o" && i + 1 < argc) output_file = argv[++i];
    }

    try {
        auto t_setup_start = std::chrono::steady_clock::now();

        // --- ONE-TIME SETUP ---
        auto device = xrt::device(0);
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        auto kernel = xrt::kernel(context, kernel_name);

        std::vector<uint32_t> instr_v = load_instr_binary(insts_path);

        auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_f = xrt::bo(device, FEATURE_COUNT * sizeof(INPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_w = xrt::bo(device, FEATURE_COUNT * sizeof(INPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_out = xrt::bo(device, 1 * sizeof(OUTPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        auto bo_tmp = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        auto bo_trace = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        bo_instr.write(instr_v.data());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Load both matrices once. Weight rows (one per class) are reused
        // across every sample, no re-reading, no re-writing per dispatch.
        auto features = load_matrix(features_file);
        auto weights = load_matrix(weights_file);

        auto t_setup_end = std::chrono::steady_clock::now();

        // --- DISPATCH LOOP (all setup already done, buffers already open) ---
        std::vector<std::vector<float>> scores(features.size(), std::vector<float>(weights.size()));

        std::vector<INPUT_TYPE> raw_f(FEATURE_COUNT), raw_w(FEATURE_COUNT);

        auto t_loop_start = std::chrono::steady_clock::now();

        for (size_t s = 0; s < features.size(); ++s) {
            for (int k = 0; k < FEATURE_COUNT; ++k) raw_f[k] = float_to_bf16(features[s][k]);
            bo_f.write(raw_f.data());
            bo_f.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            for (size_t c = 0; c < weights.size(); ++c) {
                for (int k = 0; k < FEATURE_COUNT; ++k) raw_w[k] = float_to_bf16(weights[c][k]);
                bo_w.write(raw_w.data());
                bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

                auto run = kernel(static_cast<unsigned int>(3), bo_instr, instr_v.size(),
                                   bo_f, bo_w, bo_out, bo_tmp, bo_trace);
                run.wait();

                std::vector<OUTPUT_TYPE> result(1);
                bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                bo_out.read(result.data());

                scores[s][c] = result[0];
            }
        }

        auto t_loop_end = std::chrono::steady_clock::now();

        // --- WRITE RESULTS ---
        std::ofstream out(output_file);
        for (const auto &row : scores) {
            for (size_t c = 0; c < row.size(); ++c) {
                out << row[c];
                if (c + 1 < row.size()) out << " ";
            }
            out << "\n";
        }

        double setup_ms = std::chrono::duration<double, std::milli>(t_setup_end - t_setup_start).count();
        double loop_ms = std::chrono::duration<double, std::milli>(t_loop_end - t_loop_start).count();
        int total_dispatches = (int)(features.size() * weights.size());

        std::cout << "NPU Batch Calculation Success." << std::endl;
        std::cout << "One-time setup: " << setup_ms << " ms" << std::endl;
        std::cout << "Dispatch loop total: " << loop_ms << " ms for "
                  << total_dispatches << " dispatches ("
                  << (loop_ms / total_dispatches) << " ms/dispatch avg, warm)" << std::endl;
        std::cout << "Results written to " << output_file << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}