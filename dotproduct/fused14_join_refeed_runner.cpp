// fused14_join_refeed_runner.cpp
//
// Dispatches fused14_join_refeed.xclbin: 2 transform tiles + 12 classifier
// tiles in one dispatch. The two transform outputs are joined on chip into an
// 840 value bf16 buffer, drained to the host (arg3), then fed back into the
// classifier's inF_col0/1/2 inside the same runtime sequence. This is NOT a
// fully on-chip bridge: features make one host memory hop.
//
// Arguments (matching the runtime_sequence in fused_pipeline_kernel.py):
//   arg0: 8062 int32 -- transform input, both tiles:
//         tile0 [series(251)|bias(420)|smean(420)|sscale(420)|params(2520)]
//         tile1 [same layout]
//   arg1: 840 bf16   -- classifier weights (from host)
//   arg2: 12 f32     -- classifier partial sums (4 per class column)
//   arg3: 840 bf16   -- joined transform features. Written by the device
//                       (drain of bridge_out), then read back by the device
//                       (refeed into inF_col0/1/2). Read here after dispatch
//                       so the join placement can be checked.
//
// XRT convention (same as dotprod12_runner.cpp): kernel(3, instr, size, then
// five buffer args at group_ids 3..7). Four are real, the fifth is a dummy.
//
// Build:
//   g++ -std=c++17 -o fused14_join_refeed_runner fused14_join_refeed_runner.cpp \
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -luuid
//
// Run:
//   ./fused14_join_refeed_runner -x fused14_join_refeed.xclbin \
//       -i fused14_join_refeed_sequence.bin \
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
const int FEATURE_LEN = 840;

uint16_t float_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    return static_cast<uint16_t>(i >> 16);
}

float bf16_to_float(uint16_t h) {
    uint32_t i = static_cast<uint32_t>(h) << 16;
    float f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
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
    std::string xclbin_path = "fused14_join_refeed.xclbin";
    std::string insts_path = "fused14_join_refeed_sequence.bin";
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
        auto bo_feat = xrt::bo(device, FEATURE_LEN * sizeof(uint16_t),
                                XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        auto bo_dummy = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

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

        // Zero arg3 so any stale data cannot pass for a real drain
        std::vector<uint16_t> zero_feat(FEATURE_LEN, 0);
        bo_feat.write(zero_feat.data());

        bo_instr.write(instr_v.data());
        bo_transform_in.write(raw_transform_in.data());
        bo_weights.write(raw_weights.data());
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_transform_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_weights.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_feat.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto start = std::chrono::high_resolution_clock::now();
        for (int iter = 0; iter < num_iterations; iter++) {
            auto run = kernel(static_cast<unsigned int>(3), bo_instr, instr_v.size(),
                               bo_transform_in, bo_weights, bo_out, bo_feat, bo_dummy);
            run.wait();
        }
        auto end = std::chrono::high_resolution_clock::now();
        double total_ms = std::chrono::duration<double, std::milli>(end - start).count();

        std::vector<float> partials(NUM_PARTIALS);
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(partials.data());

        std::vector<uint16_t> feat_bf16(FEATURE_LEN);
        bo_feat.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_feat.read(feat_bf16.data());

        std::cout << "\n=== PARTIAL SUMS (4 per class column) ===" << std::endl;
        for (int t = 0; t < NUM_PARTIALS; t++) {
            std::cout << "  tile " << t << ": " << partials[t] << std::endl;
        }
        std::cout << "\n=== PER-CLASS-COLUMN SCORES ===" << std::endl;
        for (int c = 0; c < 3; c++) {
            float s = 0.0f;
            for (int k = 0; k < 4; k++) s += partials[c * 4 + k];
            std::cout << "  column " << c << ": " << s << std::endl;
        }

        std::cout << "\n=== TIMING ===" << std::endl;
        std::cout << "Iterations: " << num_iterations << std::endl;
        std::cout << "Total time: " << total_ms << " ms" << std::endl;
        std::cout << "Per-dispatch: " << (total_ms / num_iterations) << " ms" << std::endl;

        std::ofstream outFile("fused14_output.txt");
        for (int i = 0; i < NUM_PARTIALS; i++) outFile << partials[i] << "\n";
        outFile.close();

        std::ofstream featFile("fused14_features_out.txt");
        int nonzero = 0;
        for (int i = 0; i < FEATURE_LEN; i++) {
            featFile << bf16_to_float(feat_bf16[i]) << "\n";
            if (feat_bf16[i] != 0) nonzero++;
        }
        featFile.close();
        std::cout << "\nDrained features written to fused14_features_out.txt ("
                  << nonzero << " of " << FEATURE_LEN << " nonzero)" << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
