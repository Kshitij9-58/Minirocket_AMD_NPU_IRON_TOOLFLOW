// transform_full420_runner.cpp
//
// Full-scale version of transform_test8_runner.cpp: FEATURES_PER_TILE=420
// (the real design). Same XRT pattern as the proven dotprod12_runner.cpp.
//
// Build:
//   g++ -std=c++17 -o transform_full420_runner transform_full420_runner.cpp \
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -luuid
//
// Run:
//   ./transform_full420_runner -x transform_full420.xclbin \
//       -i transform_full420_sequence.bin -d combined_input_full420.txt -n 100

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
const int FEATURES_PER_TILE = 420;
const int PARAMS_PER_FEATURE = 6;
const int NUM_TILES = 2;
const int COMBINED_IN_LEN = SEQ_LENGTH + 3 * FEATURES_PER_TILE + FEATURES_PER_TILE * PARAMS_PER_FEATURE;  // 4031
const int TOTAL_IN_LEN = COMBINED_IN_LEN * NUM_TILES;  // 8062
const int TOTAL_OUT_LEN = FEATURES_PER_TILE * NUM_TILES;  // 840

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
    std::string xclbin_path = "transform_full420.xclbin";
    std::string insts_path = "transform_full420_sequence.bin";
    std::string kernel_name = "MLIR_AIE";
    std::string input_file = "combined_input_full420.txt";
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
        auto bo_dummy1 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        auto bo_dummy2 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        auto bo_dummy3 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

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

        std::cout << "\n=== TIMING ===" << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "Total time: " << total_ms << " ms" << std::endl;
        std::cout << "Per-dispatch: " << (total_ms / num_iterations) << " ms" << std::endl;
        std::cout << "NPU Transform Execution Success (840 features)." << std::endl;

        std::ofstream outFile("npu_output_full420.txt");
        for (int i = 0; i < TOTAL_OUT_LEN; i++) {
            outFile << results[i] << "\n";
        }
        outFile.close();
        std::cout << "Wrote " << TOTAL_OUT_LEN << " results to npu_output_full420.txt" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
