// transform_test8_runner.cpp
//
// Dispatches the test-scale (8 features/tile) real MiniROCKET transform
// kernel and reads back its 16 output values (8 per tile x 2 tiles).
//
// Structure mirrors dotprod12_runner.cpp exactly -- same XRT setup
// sequence, same 8-argument DPU convention (opcode, instr_bo,
// instr_size, then real buffers, then dummy placeholders), same
// sync/read pattern. Only the buffer count/sizes and data types differ:
// this kernel takes ONE combined int32 input buffer and produces ONE
// int32 output buffer, versus the classifier's two bf16 inputs and one
// float output.
//
// Input buffer layout (must match the MLIR generator's convention):
//   per tile: [series(251) | bias(8) | smean(8) | sscale(8)] = 275 int32
//   full buffer: tile0's 275, then tile1's 275 = 550 int32
// All values scaled by 1000 (the project's fixed-point convention).
//
// Output: 16 int32 values (tile0's 8, then tile1's 8), each a
// normalized feature value scaled by 1000.
//
// Build:
//   g++ -std=c++17 -o transform_test8_runner transform_test8_runner.cpp \
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -luuid
//
// Run:
//   ./transform_test8_runner -x transform_test8.xclbin \
//       -i transform_test8_sequence.bin -d combined_input.txt

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <string>
#include <chrono>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using DATA_TYPE = int32_t;

const int SEQ_LENGTH = 251;
const int FEATURES_PER_TILE = 8;      // TEST_FEATURE_LIMIT
const int NUM_TILES = 2;
const int COMBINED_IN_LEN = SEQ_LENGTH + 3 * FEATURES_PER_TILE;  // 275
const int TOTAL_IN_LEN = COMBINED_IN_LEN * NUM_TILES;            // 550
const int TOTAL_OUT_LEN = FEATURES_PER_TILE * NUM_TILES;         // 16

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

int main(int argc, char *argv[]) {
    std::string xclbin_path = "transform_test8.xclbin";
    std::string insts_path = "transform_test8_sequence.bin";
    std::string kernel_name = "MLIR_AIE";
    std::string input_file = "combined_input.txt";
    int num_iterations = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) xclbin_path = argv[++i];
        else if (arg == "-i" && i + 1 < argc) insts_path = argv[++i];
        else if (arg == "-k" && i + 1 < argc) kernel_name = argv[++i];
        else if (arg == "-d" && i + 1 < argc) input_file = argv[++i];
        else if (arg == "-n" && i + 1 < argc) num_iterations = std::stoi(argv[++i]);
    }

    try {
        auto device = xrt::device(0);
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        xrt::hw_context context(device, xclbin.get_uuid());
        auto kernel = xrt::kernel(context, kernel_name);

        std::vector<uint32_t> instr_v = load_instr_binary(insts_path);

        auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t),
                                 XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
        auto bo_in = xrt::bo(device, TOTAL_IN_LEN * sizeof(DATA_TYPE),
                              XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_out = xrt::bo(device, TOTAL_OUT_LEN * sizeof(DATA_TYPE),
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        // Dummy placeholder buffers -- the DPU kernel wrapper always
        // expects 8 arguments regardless of how many the MLIR
        // runtime_sequence actually declares (established project
        // convention, see dotprod12_runner.cpp which uses bo_tmp/bo_trace
        // for the same reason).
        auto bo_dummy1 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        auto bo_dummy2 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        auto bo_dummy3 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        // Load the combined input buffer (one int32 per line, 550 values)
        std::vector<DATA_TYPE> raw_in(TOTAL_IN_LEN, 0);
        std::ifstream fileIn(input_file);
        if (!fileIn) throw std::runtime_error("Missing " + input_file);
        int count = 0;
        long long val;
        while (count < TOTAL_IN_LEN && (fileIn >> val)) {
            raw_in[count++] = static_cast<DATA_TYPE>(val);
        }
        if (count != TOTAL_IN_LEN) {
            throw std::runtime_error("Expected " + std::to_string(TOTAL_IN_LEN) +
                                      " input values, got " + std::to_string(count));
        }
        std::cout << "Loaded " << count << " input values from " << input_file << std::endl;

        bo_instr.write(instr_v.data());
        bo_in.write(raw_in.data());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Timed dispatch loop
        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; iter++) {
            auto run = kernel(static_cast<unsigned int>(3), bo_instr, instr_v.size(),
                               bo_in, bo_out, bo_dummy1, bo_dummy2, bo_dummy3);
            run.wait();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<DATA_TYPE> results(TOTAL_OUT_LEN);
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(results.data());

        std::cout << "\n=== NPU TRANSFORM OUTPUT (scaled by 1000) ===" << std::endl;
        for (int t = 0; t < NUM_TILES; t++) {
            for (int f = 0; f < FEATURES_PER_TILE; f++) {
                int idx = t * FEATURES_PER_TILE + f;
                std::cout << "  tile" << t << " feat" << f << ": "
                          << results[idx] << "  (= " << (results[idx] / 1000.0) << ")" << std::endl;
            }
        }

        std::cout << "\n=== TIMING ===" << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "Total time: " << total_ms << " ms" << std::endl;
        std::cout << "Per-dispatch: " << (total_ms / num_iterations) << " ms" << std::endl;
        std::cout << "\nNPU Transform Execution Success." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
