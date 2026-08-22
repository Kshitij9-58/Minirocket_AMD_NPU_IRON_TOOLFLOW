// Minimal XRT timing test: measures just the open/launch/wait/sync cycle
// with the smallest possible payload (a single zero-value dispatch),
// to isolate fixed per-dispatch XRT overhead from data transfer cost.
#include <iostream>
#include <fstream>
#include <vector>
#include <cstring>
#include <chrono>
#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using INPUT_TYPE = uint16_t;
using OUTPUT_TYPE = float;
const int FEATURE_COUNT = 840;

std::vector<uint32_t> load_instr(const std::string &f) {
    std::ifstream file(f, std::ios::binary | std::ios::ate);
    std::streamsize size = file.tellg(); file.seekg(0);
    std::vector<uint32_t> buf(size/4);
    file.read((char*)buf.data(), size);
    return buf;
}

int main() {
    std::string xclbin_path = "dotproduct.xclbin";
    std::string insts_path = "dotproduct_insts.txt";
    std::string kernel_name = "MLIR_AIE";

    auto device = xrt::device(0);
    auto xclbin = xrt::xclbin(xclbin_path);
    device.register_xclbin(xclbin);
    xrt::hw_context ctx(device, xclbin.get_uuid());
    auto kernel = xrt::kernel(ctx, kernel_name);
    auto instr_v = load_instr(insts_path);

    auto bo_instr = xrt::bo(device, instr_v.size()*4, XCL_BO_FLAGS_CACHEABLE, kernel.group_id(1));
    auto bo_f  = xrt::bo(device, FEATURE_COUNT*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(3));
    auto bo_w  = xrt::bo(device, FEATURE_COUNT*2, XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(4));
    auto bo_out= xrt::bo(device, 4,               XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(5));
    auto bo_tmp= xrt::bo(device, 4096,            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(6));
    auto bo_tr = xrt::bo(device, 4096,            XRT_BO_FLAGS_HOST_ONLY, kernel.group_id(7));

    bo_instr.write(instr_v.data());
    bo_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    std::vector<INPUT_TYPE> zeros(FEATURE_COUNT, 0);
    bo_f.write(zeros.data()); bo_f.sync(XCL_BO_SYNC_BO_TO_DEVICE);
    bo_w.write(zeros.data()); bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);

    // Warmup
    for (int i = 0; i < 5; i++) {
        auto run = kernel((unsigned)3, bo_instr, instr_v.size(), bo_f, bo_w, bo_out, bo_tmp, bo_tr);
        run.wait();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }

    // Timed loop: no data changes, just pure dispatch/wait/sync cost
    const int N = 100;
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        auto run = kernel((unsigned)3, bo_instr, instr_v.size(), bo_f, bo_w, bo_out, bo_tmp, bo_tr);
        run.wait();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
    auto t1 = std::chrono::steady_clock::now();
    double ms = std::chrono::duration<double,std::milli>(t1-t0).count();
    std::cout << "Pure dispatch/wait/sync (no data change): "
              << ms/N << " ms/dispatch over " << N << " iterations" << std::endl;

    // Timed loop: WITH full H2D sync each iteration (simulates real dispatch)
    auto t2 = std::chrono::steady_clock::now();
    for (int i = 0; i < N; i++) {
        bo_f.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        bo_w.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        auto run = kernel((unsigned)3, bo_instr, instr_v.size(), bo_f, bo_w, bo_out, bo_tmp, bo_tr);
        run.wait();
        bo_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
    }
    auto t3 = std::chrono::steady_clock::now();
    double ms2 = std::chrono::duration<double,std::milli>(t3-t2).count();
    std::cout << "Full H2D+dispatch+D2H (real dispatch cost): "
              << ms2/N << " ms/dispatch over " << N << " iterations" << std::endl;

    std::cout << "DMA transfer cost alone: " << (ms2-ms)/N << " ms/dispatch" << std::endl;
    return 0;
}