#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>
#include <string>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using INPUT_TYPE = uint16_t;
using OUTPUT_TYPE = float;

const int FEATURE_COUNT = 840;
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
    std::string xclbin_path = "main.xclbin";
    std::string insts_path = "main_sequence.bin";
    std::string kernel_name = "MLIR_AIE";
    std::string features_file = "features.txt";
    std::string weights_file = "weights.txt";

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "-x" && i + 1 < argc) xclbin_path = argv[++i];
        else if (arg == "-i" && i + 1 < argc) insts_path = argv[++i];
        else if (arg == "-k" && i + 1 < argc) kernel_name = argv[++i];
        else if (arg == "-f" && i + 1 < argc) features_file = argv[++i];
        else if (arg == "-w" && i + 1 < argc) weights_file = argv[++i];
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
        auto bo_f = xrt::bo(device, FEATURE_COUNT * sizeof(INPUT_TYPE),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
        auto bo_w = xrt::bo(device, FEATURE_COUNT * sizeof(INPUT_TYPE),
                             XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
        auto bo_out = xrt::bo(device, NUM_PARTIALS * sizeof(OUTPUT_TYPE),
                               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
        auto bo_tmp = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
        auto bo_trace = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

        std::vector<INPUT_TYPE> raw_F(FEATURE_COUNT), raw_W(FEATURE_COUNT);
        float val_f;

        std::ifstream fileF(features_file);
        if (!fileF) throw std::runtime_error("Missing " + features_file);
        for (int i = 0; i < FEATURE_COUNT && (fileF >> val_f); ++i) raw_F[i] = float_to_bf16(val_f);

        std::ifstream fileW(weights_file);
        if (!fileW) throw std::runtime_error("Missing " + weights_file);
        for (int i = 0; i < FEATURE_COUNT && (fileW >> val_f); ++i) raw_W[i] = float_to_bf16(val_f);

        bo_instr.write(instr_v.data());
        bo_f.write(raw_F.data());
        bo_w.write(raw_W.data());

        bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_f.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

        auto run = kernel(static_cast<unsigned int>(3), bo_instr, instr_v.size(),
                           bo_f, bo_w, bo_out, bo_tmp, bo_trace);
        run.wait();

        std::vector<OUTPUT_TYPE> partials(NUM_PARTIALS);
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
        bo_out.read(partials.data());

        float final_score = 0.0f;
        std::cout << "Per-tile partial sums:" << std::endl;
        for (int t = 0; t < NUM_PARTIALS; t++) {
            std::cout << "  tile " << t << ": " << partials[t] << std::endl;
            final_score += partials[t];
        }

        std::cout << "NPU Calculation Success." << std::endl;
        std::cout << "Dot Product Score: " << final_score << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
