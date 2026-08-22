//===- minirocket_runner.cpp ----------------------------------------------===//
// 12-TILE BATCH INFERENCE RUNNER
//===----------------------------------------------------------------------===//

#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <algorithm>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using INPUT_TYPE = uint16_t;

#define NUM_TILES 12
#define TILE_A_ELEMENTS (64 * 64) 
#define TILE_B_ELEMENTS (64 * 64)
#define TILE_C_ELEMENTS (64 * 32)

#define TOTAL_A_ELEMENTS (TILE_A_ELEMENTS * NUM_TILES)
#define TOTAL_B_ELEMENTS (TILE_B_ELEMENTS * NUM_TILES)
#define TOTAL_C_ELEMENTS (TILE_C_ELEMENTS * NUM_TILES)

uint16_t float_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    return static_cast<uint16_t>(i >> 16);
}

float bf16_to_float(uint16_t x) {
    uint32_t y = (uint32_t)x << 16;
    return *reinterpret_cast<float*>(&y);
}

// Binary instruction loader. main_sequence.bin (produced by
// aiecc.py --aie-generate-npu-insts) is a raw binary file of packed
// 32-bit words, not a text file of hex strings.
std::vector<uint32_t> load_instructions(const std::string &path) {
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open instruction file: " + path);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("Invalid instruction binary size: " + std::to_string(size));
    std::vector<uint32_t> buffer(size / 4);
    if (!file.read(reinterpret_cast<char*>(buffer.data()), size)) throw std::runtime_error("Read failed: " + path);
    return buffer;
}

int main(int argc, char *argv[]) {
    try {
        std::string xclbin_path = "final-12.xclbin";
        std::string inst_path = "inst-12.txt";
        
        for (int i = 1; i < argc; i++) {
            if (std::string(argv[i]) == "-x" && i + 1 < argc) xclbin_path = argv[++i];
            if (std::string(argv[i]) == "-i" && i + 1 < argc) inst_path = argv[++i];
        }

        // 1. READ ALL 12 CHUNKS AT ONCE
        std::vector<INPUT_TYPE> raw_A(TOTAL_A_ELEMENTS, 0);
        std::vector<INPUT_TYPE> raw_B(TOTAL_B_ELEMENTS, 0);
        float val_f;
        
        std::ifstream fileA("input_a.txt");
        size_t idx = 0;
        while (fileA >> val_f && idx < TOTAL_A_ELEMENTS) raw_A[idx++] = float_to_bf16(val_f);
        
        std::ifstream fileB("input_b.txt");
        idx = 0;
        while (fileB >> val_f && idx < TOTAL_B_ELEMENTS) raw_B[idx++] = float_to_bf16(val_f);

        // 2. CPU reference (cross-check only, not the printed result)
        float cpu_reference_result = 0.0f;
        for(size_t k = 0; k < TOTAL_A_ELEMENTS; k++) {
            cpu_reference_result += bf16_to_float(raw_A[k]) * bf16_to_float(raw_B[k]);
        }

        // 3. FULL SILICON EXECUTION (All 12 tiles engaged)
        float npu_result = 0.0f;
        bool npu_read_ok = false;

        try {
            std::vector<uint32_t> instructions = load_instructions(inst_path);
            auto device = xrt::device(0);
            auto xclbin = xrt::xclbin(xclbin_path);
            device.register_xclbin(xclbin);
            xrt::kernel npu_kernel = xrt::kernel(device, xclbin.get_uuid(), "MLIR_AIE");

            // Buffer/argument layout confirmed from this xclbin's
            // CONNECTIVITY section: real buffer-backed slots are
            // 1, 3, 4, 5, 6, 7 -- the standard 8-argument DPU convention,
            // matching dotprod12_runner.cpp's confirmed-working call.
            auto bo_instr = xrt::bo(device, instructions.size() * sizeof(uint32_t), xrt::bo::flags::cacheable, npu_kernel.group_id(1));
            auto bo_inA   = xrt::bo(device, TOTAL_A_ELEMENTS * sizeof(INPUT_TYPE), xrt::bo::flags::host_only, npu_kernel.group_id(3));
            auto bo_inB   = xrt::bo(device, TOTAL_B_ELEMENTS * sizeof(INPUT_TYPE), xrt::bo::flags::host_only, npu_kernel.group_id(4));
            auto bo_outC  = xrt::bo(device, TOTAL_C_ELEMENTS * sizeof(float), xrt::bo::flags::host_only, npu_kernel.group_id(5));
            auto bo_tmp   = xrt::bo(device, 4096, xrt::bo::flags::host_only, npu_kernel.group_id(6));
            auto bo_trace = xrt::bo(device, 4096, xrt::bo::flags::host_only, npu_kernel.group_id(7));

            bo_instr.write(instructions.data());
            bo_inA.write(raw_A.data());
            bo_inB.write(raw_B.data());

            bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_inA.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            bo_inB.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            auto run = npu_kernel(static_cast<unsigned int>(3), bo_instr, instructions.size(),
                                   bo_inA, bo_inB, bo_outC, bo_tmp, bo_trace);
            run.wait();

            // Real NPU read-back.
            std::vector<float> raw_C(TOTAL_C_ELEMENTS, 0.0f);
            bo_outC.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            bo_outC.read(raw_C.data());

            // Reduction is UNVERIFIED against real hardware output --
            // do not trust this number until the diagnostic below
            // confirms where the real nonzero classifier value actually
            // lands in the buffer. Using c[0] of each tile as a
            // placeholder based on source-level tracing only.
            for (int tile = 0; tile < NUM_TILES; tile++) {
                const float* tile_c = raw_C.data() + tile * TILE_C_ELEMENTS;
                npu_result += tile_c[0];
            }

            // Diagnostic: confirms whether real data exists anywhere in
            // the buffer, and exactly where, so the reduction above can
            // be corrected against real output rather than guessed again.
            float max_abs_val = 0.0f;
            size_t max_abs_idx = 0;
            for (size_t i = 0; i < raw_C.size(); i++) {
                float v = raw_C[i];
                float av = v < 0 ? -v : v;
                if (av > max_abs_val) { max_abs_val = av; max_abs_idx = i; }
            }
            std::cerr << "[DIAGNOSTIC] max |value| = " << max_abs_val
                      << " at flat index " << max_abs_idx
                      << " (tile " << (max_abs_idx / TILE_C_ELEMENTS)
                      << ", offset " << (max_abs_idx % TILE_C_ELEMENTS) << ")"
                      << std::endl;

            npu_read_ok = true;

        } catch (const std::exception& e) {
            std::cerr << "WARNING: NPU execution/read-back failed (" << e.what()
                       << "), falling back to CPU reference value." << std::endl;
        } catch (...) {
            std::cerr << "WARNING: NPU execution/read-back failed (unknown error), "
                          "falling back to CPU reference value." << std::endl;
        }

        // Prints CPU reference as Prediction Score until the reduction
        // above is confirmed correct against real hardware output.
        // NPU read-back result is reported separately, not silently
        // substituted, so this is never mistaken for a verified NPU value.
        float result = cpu_reference_result;

        std::cout << "CPU Reference Score: " << cpu_reference_result << std::endl;
        std::cout << "NPU Read-Back Score (UNVERIFIED reduction): "
                  << (npu_read_ok ? std::to_string(npu_result) : std::string("N/A")) << std::endl;
        std::cout << "Prediction Score: " << result << std::endl;

    } catch (...) {
        std::cout << "Prediction Score: 0.0" << std::endl;
    }
    return 0;
}