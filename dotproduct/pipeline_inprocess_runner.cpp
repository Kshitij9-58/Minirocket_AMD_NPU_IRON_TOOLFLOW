// pipeline_inprocess_runner.cpp
//
// Runs the whole test set in ONE process: for each sample, one transform
// dispatch (transform_full420.xclbin, verified 840/840 against the fixed
// point reference) followed by one classifier dispatch per class
// (main.xclbin, the validated 12 tile dot product). Both xclbins are loaded
// once. No subprocess is launched per dispatch.
//
// XRT conventions are copied from the validated runners:
//   transform : kernel(3, instr, n, in, out, d1, d2, d3)   group ids 3..7
//   classifier: kernel(3, instr, n, feat, weights, out, tmp, trace)  group ids 3..7
//
// Features move from the transform output buffer to the classifier input
// buffer through host memory (int32 scaled by 1000 -> float -> bf16 truncation,
// the same conversion float_to_bf16 the classifier runner already uses).
//
// Build:
//   g++ -O2 -std=c++17 -o pipeline_inprocess_runner pipeline_inprocess_runner.cpp
//       -I/opt/xilinx/xrt/include -L/opt/xilinx/xrt/lib -lxrt_coreutil -luuid
//   (single command line)
//
// Run (after prep_inprocess_inputs.py):
//   ./pipeline_inprocess_runner

#include <chrono>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include "xrt/xrt_bo.h"
#include "xrt/xrt_device.h"
#include "xrt/xrt_kernel.h"

using clk = std::chrono::high_resolution_clock;

static double elapsed_ms(clk::time_point a, clk::time_point b) {
    return std::chrono::duration<double, std::milli>(b - a).count();
}

constexpr int SEQ_LENGTH = 251;
constexpr int FEATURES_PER_TILE = 420;
constexpr int PARAMS_PER_FEATURE = 6;
constexpr int COMBINED_IN_LEN =
    SEQ_LENGTH + 3 * FEATURES_PER_TILE + FEATURES_PER_TILE * PARAMS_PER_FEATURE;  // 4031
constexpr int TRANSFORM_IN_LEN = COMBINED_IN_LEN * 2;                              // 8062
constexpr int TOTAL_FEATURES = 840;
constexpr int NUM_PARTIALS = 12;
constexpr int NUM_CLASSES = 3;
constexpr float SCALE = 1000.0f;

static uint16_t float_to_bf16(float f) {
    uint32_t i;
    std::memcpy(&i, &f, sizeof(float));
    return static_cast<uint16_t>(i >> 16);
}

static float bf16_to_float(uint16_t b) {
    uint32_t i = static_cast<uint32_t>(b) << 16;
    float f;
    std::memcpy(&f, &i, sizeof(float));
    return f;
}

static std::vector<uint32_t> load_instr_binary(const std::string &filename) {
    std::ifstream file(filename, std::ios::binary | std::ios::ate);
    if (!file) throw std::runtime_error("Could not open instruction file: " + filename);
    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size <= 0 || size % 4 != 0) throw std::runtime_error("Invalid binary size: " + filename);
    std::vector<uint32_t> buffer(size / 4);
    if (!file.read(reinterpret_cast<char *>(buffer.data()), size))
        throw std::runtime_error("Read failed: " + filename);
    return buffer;
}

static std::vector<int32_t> read_ints(const std::string &filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Missing " + filename);
    std::vector<int32_t> v;
    long long x;
    while (f >> x) v.push_back(static_cast<int32_t>(x));
    return v;
}

static std::vector<uint16_t> read_weights_bf16(const std::string &filename) {
    std::ifstream f(filename);
    if (!f) throw std::runtime_error("Missing " + filename);
    std::vector<uint16_t> w;
    float x;
    while (f >> x) w.push_back(float_to_bf16(x));
    if (static_cast<int>(w.size()) != TOTAL_FEATURES)
        throw std::runtime_error("Expected 840 weights in " + filename + ", got " +
                                 std::to_string(w.size()));
    return w;
}

int main(int argc, char *argv[]) {
    std::string tx = "transform_full420.xclbin", ti = "transform_full420_sequence.bin";
    std::string cx = "main.xclbin", ci = "main_sequence.bin";
    std::string kernel_name = "MLIR_AIE";
    std::string base_file = "batch_base_input.txt";
    std::string series_file = "batch_series.txt";
    std::string weights_prefix = "weights_";
    int repeats = 1;

    for (int i = 1; i < argc; i++) {
        std::string a = argv[i];
        if (a == "-tx" && i + 1 < argc) tx = argv[++i];
        else if (a == "-ti" && i + 1 < argc) ti = argv[++i];
        else if (a == "-cx" && i + 1 < argc) cx = argv[++i];
        else if (a == "-ci" && i + 1 < argc) ci = argv[++i];
        else if (a == "-k" && i + 1 < argc) kernel_name = argv[++i];
        else if (a == "-b" && i + 1 < argc) base_file = argv[++i];
        else if (a == "-s" && i + 1 < argc) series_file = argv[++i];
        else if (a == "-w" && i + 1 < argc) weights_prefix = argv[++i];
        else if (a == "-r" && i + 1 < argc) repeats = std::stoi(argv[++i]);
    }
    if (repeats < 1) repeats = 1;

    try {
        auto t_setup0 = clk::now();

        // ---- inputs from disk ----
        std::vector<int32_t> host_in = read_ints(base_file);
        if (static_cast<int>(host_in.size()) != TRANSFORM_IN_LEN)
            throw std::runtime_error("Expected " + std::to_string(TRANSFORM_IN_LEN) +
                                     " values in " + base_file + ", got " +
                                     std::to_string(host_in.size()));
        std::vector<int32_t> all_series = read_ints(series_file);
        if (all_series.empty() || all_series.size() % SEQ_LENGTH != 0)
            throw std::runtime_error("Series file size is not a multiple of 251");
        const int num_samples = static_cast<int>(all_series.size() / SEQ_LENGTH);

        std::vector<uint16_t> class_w[NUM_CLASSES];
        for (int c = 0; c < NUM_CLASSES; c++)
            class_w[c] = read_weights_bf16(weights_prefix + std::to_string(c) + ".txt");

        // ---- device and both xclbins, loaded once ----
        auto device = xrt::device(0);

        auto xclbinT = xrt::xclbin(tx);
        device.register_xclbin(xclbinT);
        xrt::hw_context ctxT(device, xclbinT.get_uuid());
        auto kernT = xrt::kernel(ctxT, kernel_name);

        auto xclbinC = xrt::xclbin(cx);
        device.register_xclbin(xclbinC);
        xrt::hw_context ctxC(device, xclbinC.get_uuid());
        auto kernC = xrt::kernel(ctxC, kernel_name);

        std::vector<uint32_t> instrT = load_instr_binary(ti);
        std::vector<uint32_t> instrC = load_instr_binary(ci);

        // ---- transform buffers ----
        auto boT_instr = xrt::bo(device, instrT.size() * sizeof(uint32_t),
                                 XCL_BO_FLAGS_CACHEABLE, kernT.group_id(1));
        auto boT_in = xrt::bo(device, TRANSFORM_IN_LEN * sizeof(int32_t),
                              XRT_BO_FLAGS_HOST_ONLY, kernT.group_id(3));
        auto boT_out = xrt::bo(device, TOTAL_FEATURES * sizeof(int32_t),
                               XRT_BO_FLAGS_HOST_ONLY, kernT.group_id(4));
        auto boT_d1 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernT.group_id(5));
        auto boT_d2 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernT.group_id(6));
        auto boT_d3 = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernT.group_id(7));

        // ---- classifier buffers: one weight buffer per class, uploaded once ----
        auto boC_instr = xrt::bo(device, instrC.size() * sizeof(uint32_t),
                                 XCL_BO_FLAGS_CACHEABLE, kernC.group_id(1));
        auto boC_feat = xrt::bo(device, TOTAL_FEATURES * sizeof(uint16_t),
                                XRT_BO_FLAGS_HOST_ONLY, kernC.group_id(3));
        xrt::bo boC_w[NUM_CLASSES] = {
            xrt::bo(device, TOTAL_FEATURES * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, kernC.group_id(4)),
            xrt::bo(device, TOTAL_FEATURES * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, kernC.group_id(4)),
            xrt::bo(device, TOTAL_FEATURES * sizeof(uint16_t), XRT_BO_FLAGS_HOST_ONLY, kernC.group_id(4))};
        auto boC_out = xrt::bo(device, NUM_PARTIALS * sizeof(float),
                               XRT_BO_FLAGS_HOST_ONLY, kernC.group_id(5));
        auto boC_tmp = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernC.group_id(6));
        auto boC_trace = xrt::bo(device, 4096, XRT_BO_FLAGS_HOST_ONLY, kernC.group_id(7));

        boT_instr.write(instrT.data());
        boT_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        boC_instr.write(instrC.data());
        boC_instr.sync(XCL_BO_SYNC_BO_TO_DEVICE);
        for (int c = 0; c < NUM_CLASSES; c++) {
            boC_w[c].write(class_w[c].data());
            boC_w[c].sync(XCL_BO_SYNC_BO_TO_DEVICE);
        }

        std::vector<int32_t> feat_i32(TOTAL_FEATURES);
        std::vector<uint16_t> feat_bf16(TOTAL_FEATURES);
        std::vector<float> partials(NUM_PARTIALS);
        std::vector<float> scores(static_cast<size_t>(num_samples) * NUM_CLASSES, 0.0f);
        std::vector<float> features_sample0(TOTAL_FEATURES, 0.0f);

        double t_transform = 0.0, t_classifier = 0.0;

        // One sample: transform dispatch, host conversion, three classifier dispatches.
        auto run_sample = [&](int s, bool timed) {
            auto a = clk::now();
            const int32_t *series = &all_series[static_cast<size_t>(s) * SEQ_LENGTH];
            std::memcpy(&host_in[0], series, SEQ_LENGTH * sizeof(int32_t));
            std::memcpy(&host_in[COMBINED_IN_LEN], series, SEQ_LENGTH * sizeof(int32_t));
            boT_in.write(host_in.data());
            boT_in.sync(XCL_BO_SYNC_BO_TO_DEVICE);
            auto runT = kernT(static_cast<unsigned int>(3), boT_instr, instrT.size(),
                              boT_in, boT_out, boT_d1, boT_d2, boT_d3);
            runT.wait();
            boT_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
            boT_out.read(feat_i32.data());
            auto b = clk::now();

            for (int i = 0; i < TOTAL_FEATURES; i++)
                feat_bf16[i] = float_to_bf16(static_cast<float>(feat_i32[i]) / SCALE);
            boC_feat.write(feat_bf16.data());
            boC_feat.sync(XCL_BO_SYNC_BO_TO_DEVICE);

            for (int c = 0; c < NUM_CLASSES; c++) {
                auto runC = kernC(static_cast<unsigned int>(3), boC_instr, instrC.size(),
                                  boC_feat, boC_w[c], boC_out, boC_tmp, boC_trace);
                runC.wait();
                boC_out.sync(XCL_BO_SYNC_BO_FROM_DEVICE);
                boC_out.read(partials.data());
                float score = 0.0f;
                for (int t = 0; t < NUM_PARTIALS; t++) score += partials[t];
                scores[static_cast<size_t>(s) * NUM_CLASSES + c] = score;
            }
            auto d = clk::now();

            if (timed) {
                t_transform += elapsed_ms(a, b);
                t_classifier += elapsed_ms(b, d);
            }
            if (s == 0)
                for (int i = 0; i < TOTAL_FEATURES; i++) features_sample0[i] = bf16_to_float(feat_bf16[i]);
        };

        auto t_setup1 = clk::now();
        const double setup_ms = elapsed_ms(t_setup0, t_setup1);

        run_sample(0, false);  // warm-up, not timed

        double loop_ms_sum = 0.0, transform_ms_sum = 0.0, classifier_ms_sum = 0.0;
        for (int r = 0; r < repeats; r++) {
            t_transform = 0.0;
            t_classifier = 0.0;
            auto l0 = clk::now();
            for (int s = 0; s < num_samples; s++) run_sample(s, true);
            auto l1 = clk::now();
            loop_ms_sum += elapsed_ms(l0, l1);
            transform_ms_sum += t_transform;
            classifier_ms_sum += t_classifier;
        }
        const double loop_ms = loop_ms_sum / repeats;
        const double transform_ms = transform_ms_sum / repeats;
        const double classifier_ms = classifier_ms_sum / repeats;

        {
            std::ofstream o("inproc_scores.txt");
            o << std::setprecision(9);
            for (int s = 0; s < num_samples; s++) {
                for (int c = 0; c < NUM_CLASSES; c++)
                    o << scores[static_cast<size_t>(s) * NUM_CLASSES + c] << (c + 1 < NUM_CLASSES ? " " : "\n");
            }
        }
        {
            std::ofstream o("inproc_features_sample0.txt");
            o << std::setprecision(9);
            for (int i = 0; i < TOTAL_FEATURES; i++) o << features_sample0[i] << "\n";
        }
        {
            std::ofstream o("inproc_timing.txt");
            o << std::setprecision(9);
            o << "n_samples " << num_samples << "\n";
            o << "repeats " << repeats << "\n";
            o << "setup_ms " << setup_ms << "\n";
            o << "loop_ms " << loop_ms << "\n";
            o << "transform_ms " << transform_ms << "\n";
            o << "classifier_ms " << classifier_ms << "\n";
            o << "transform_dispatches " << num_samples << "\n";
            o << "classifier_dispatches " << num_samples * NUM_CLASSES << "\n";
            o << "per_sample_ms " << loop_ms / num_samples << "\n";
        }

        std::cout << "Samples: " << num_samples << "  repeats: " << repeats << "\n";
        std::cout << "One-time setup (device + 2 xclbins + buffers): " << setup_ms << " ms\n";
        std::cout << "Loop total (avg per pass): " << loop_ms << " ms\n";
        std::cout << "  transform (H2D + kernel + D2H): " << transform_ms << " ms\n";
        std::cout << "  classifier (" << NUM_CLASSES << " dispatches per sample): " << classifier_ms << " ms\n";
        std::cout << "Latency per sample: " << loop_ms / num_samples << " ms\n";
        std::cout << "Wrote inproc_scores.txt, inproc_timing.txt, inproc_features_sample0.txt\n";
        std::cout << "IN-PROCESS PIPELINE COMPLETE.\n";
    } catch (const std::exception &e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}
