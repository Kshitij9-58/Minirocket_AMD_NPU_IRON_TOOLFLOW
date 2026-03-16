//===- minirocket_runner.cpp ----------------------------------------------===//
// UPDATED: BF16 Configuration (64x64 Kernel Compatible)
// Input:  uint16_t (Representing bfloat16)
// Output: float (Standard 32-bit Float)
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

// --- BF16 NPU DATATYPES ---
// uint16_t is used to hold the bits of a bfloat16
using INPUT_TYPE = uint16_t;
// The NPU output for this kernel is standard float
using OUTPUT_TYPE = float;

// MATCHING THE KERNEL DIMENSIONS
const int KERNEL_ROWS = 64;
const int KERNEL_COLS = 32;
const int VOLUME_A = 64 * 64;
const int VOLUME_B = 64 * 32;
const int VOLUME_C = 64 * 32;

// Helper to convert float to bf16 bits (Truncation)
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
    // Path to the successful build artifacts
    std::string xclbin_path = "final.xclbin";
    std::string insts_path = "insts.txt"; 
    std::string kernel_name = "npu_kernel"; // Default name for many AIE-ML flows
    
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) xclbin_path = argv[++i];
        else if (arg == "-i" && i + 1 < argc) insts_path = argv[++i];
        else if (arg == "-k" && i + 1 < argc) kernel_name = argv[++i];
    }

    try {
        auto device = xrt::device(0);
        auto xclbin = xrt::xclbin(xclbin_path);
        device.register_xclbin(xclbin);
        auto uuid = device.get_uuid();
        xrt::hw_context context(device, uuid);
        auto kernel = xrt::kernel(context, kernel_name);
        
        std::vector<uint32_t> instr_v = load_instr_binary(insts_path);
        
        // --- BUFFER SETUP (BF16/FLOAT) ---
        auto bo_instr = xrt::bo(device, instr_v.size() * sizeof(uint32_t), XCL_BO_FLAGS_CACHEABLE, kernel.group_id(0));
        auto bo_a = xrt::bo(device, VOLUME_A * sizeof(INPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(1));
        auto bo_b = xrt::bo(device, VOLUME_B * sizeof(INPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(2));
        auto bo_out = xrt::bo(device, VOLUME_C * sizeof(OUTPUT_TYPE), XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));

        // --- PREPARE DATA ---
        std::vector<INPUT_TYPE> raw_A(VOLUME_A), raw_B(VOLUME_B);
        float val_f;
        
        std::ifstream fileA("input_a.txt");
        if (!fileA) throw std::runtime_error("Missing input_a.txt");
        for(int i=0; i<VOLUME_A && (fileA >> val_f); ++i) raw_A[i] = float_to_bf16(val_f);
        
        std::ifstream fileB("input_b.txt");
        if (!fileB) throw std::runtime_error("Missing input_b.txt");
        for(int i=0; i<VOLUME_B && (fileB >> val_f); ++i) raw_B[i] = float_to_bf16(val_f);

        // --- EXECUTION ---
        bo_instr.write(instr_v.data());
        bo_a.write(raw_A.data()); 
        bo_b.write(raw_B.data());
        
        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_a.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_b.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        // Trigger NPU
        auto run = kernel(bo_instr, instr_v.size(), bo_a, bo_b, bo_out);
        run.wait();

        // --- READ RESULTS ---
        std::vector<OUTPUT_TYPE> result(VOLUME_C);
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(result.data());

        std::cout << "NPU Calculation Success." << std::endl;
        std::cout << "First Prediction Score: " << result[0] << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}