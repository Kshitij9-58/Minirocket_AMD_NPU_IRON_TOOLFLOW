// fused_pipeline_runner.cpp
//
// Dispatches the fused 14-tile pipeline (2 transform tiles + 12
// classifier tiles, on-chip bridge, no host round-trip for features).
// Same XRT pattern as the proven dotprod12_runner.cpp: device/xclbin/
// kernel setup, group_id buffer allocation, sync+dispatch+read.
//
// Arguments (matching the runtime_sequence in fused_pipeline_kernel.py):
//   arg0: 8062 int32 -- transform's combined input (both tiles):
//         tile0 [series(251)|bias(420)|smean(420)|sscale(420)|params(2520)]
//         tile1 [same layout, second half of the input]
//   arg1: 840 bf16 -- classifier weights (Ridge coefficients), UNCHANGED
//         from the standalone classifier's host-fed input, still needed
//         since weights are model coefficients, not transform output
//   arg2: 12 f32 -- classifier's per-tile partial dot-product sums,
//         summed on the host exactly as dotprod12_runner.cpp already
//         does (host-side summation is the established, unchanged design)
//
// Build:
//   g++ -std=c++17 -o fused_pipeline_runner fused_pipeline_runner.cpp \
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -luuid
//
// Run:
//   ./fused_pipeline_runner -x fused_pipeline_14tile.xclbin \
//       -i fused_pipeline_14tile_sequence.bin \
//       -d combined_transform_input.txt -w classifier_weights.txt

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

const int TRANSFORM_IN_LEN = 8062;
const int CLASSIFIER_WEIGHT_LEN = 840;
const int NUM_PARTIALS = 12;

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

int main(int argc, char *argv[]) {
    std::string xclbin_path = "fused_pipeline_14tile.xclbin";
    std::string insts_path = "fused_pipeline_14tile_sequence.bin";
    std::string kernel_name = "MLIR_AIE";
    std::string transform_input_file = "combined_transform_input.txt";
    std::string weights_file = "classifier_weights.txt";
    int num_iterations = 1;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) xclbin_path = argv[++i];
        else if (arg == "-i" && i + 1 < argc) insts_path = argv[++i];
        else if (arg == "-k" && i + 1 < argc) kernel_name = argv[++i];
        else if (arg == "-d" && i + 1 < argc) transform_input_file = argv[++i];
        else if (arg == "-w" && i + 1 < argc) weights_file = argv[++i];
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
        auto bo_transform_in = xrt::bo(device, TRANSFORM_IN_LEN * sizeof(int32_t),
                                        XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_weights = xrt::bo(device, CLASSIFIER_WEIGHT_LEN * sizeof(uint16_t),
                                   XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_out = xrt::bo(device, NUM_PARTIALS * sizeof(float),
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        auto bo_dummy1 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        auto bo_dummy2 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        // Load transform combined input (int32 values, one per line)
        std::vector<int32_t> raw_transform_in(TRANSFORM_IN_LEN, 0);
        std::ifstream fileIn(transform_input_file);
        if (!fileIn) throw std::runtime_error("Missing " + transform_input_file);
        int count = 0;
        long long val;
        while (count < TRANSFORM_IN_LEN && (fileIn >> val)) {
            raw_transform_in[count++] = static_cast<int32_t>(val);
        }
        if (count != TRANSFORM_IN_LEN) {
            throw std::runtime_error("Expected " + std::to_string(TRANSFORM_IN_LEN) +
                                      " transform input values, got " + std::to_string(count));
        }
        std::cout << "Loaded " << count << " transform input values from " << transform_input_file << std::endl;

        // Load classifier weights (plain floats, converted to bf16 -- same
        // convention as dotprod12_runner.cpp)
        std::vector<uint16_t> raw_weights(CLASSIFIER_WEIGHT_LEN);
        std::ifstream fileW(weights_file);
        if (!fileW) throw std::runtime_error("Missing " + weights_file);
        float val_f;
        int wcount = 0;
        for (int i = 0; i < CLASSIFIER_WEIGHT_LEN && (fileW >> val_f); ++i) {
            raw_weights[i] = float_to_bf16(val_f);
            wcount++;
        }
        if (wcount != CLASSIFIER_WEIGHT_LEN) {
            throw std::runtime_error("Expected " + std::to_string(CLASSIFIER_WEIGHT_LEN) +
                                      " weight values, got " + std::to_string(wcount));
        }
        std::cout << "Loaded " << wcount << " weight values from " << weights_file << std::endl;

        bo_instr.write(instr_v.data());
        bo_transform_in.write(raw_transform_in.data());
        bo_weights.write(raw_weights.data());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_transform_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; iter++) {
            auto run = kernel(static_cast<unsigned int>(3), bo_instr, instr_v.size(),
                               bo_transform_in, bo_weights, bo_out, bo_dummy1, bo_dummy2);
            run.wait();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<float> partials(NUM_PARTIALS);
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(partials.data());

        float final_score = 0.0f;
        std::cout << "\n=== FUSED PIPELINE OUTPUT (per-tile partial sums) ===" << std::endl;
        for (int t = 0; t < NUM_PARTIALS; t++) {
            std::cout << "  tile " << t << ": " << partials[t] << std::endl;
            final_score += partials[t];
        }

        std::cout << "\nFinal dot-product score: " << final_score << std::endl;

        std::cout << "\n=== TIMING ===" << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "Total time: " << total_ms << " ms" << std::endl;
        std::cout << "Per-dispatch: " << (total_ms / num_iterations) << " ms" << std::endl;
        std::cout << "\nFUSED PIPELINE EXECUTION SUCCESS." << std::endl;

        std::ofstream outFile("fused_pipeline_output.txt");
        for (int i = 0; i < NUM_PARTIALS; i++) {
            outFile << partials[i] << "\n";
        }
        outFile.close();

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
