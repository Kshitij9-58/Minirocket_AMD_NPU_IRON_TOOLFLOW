// dotproduct_bf16_f32.cc
// Minimal single-tile dot-product microkernel for the classifier.
// score = sum(features[i] * weights[i]) for i in [0, FEATURE_COUNT)
//
// STATUS: untested starting point, not yet compiled or run. Calling
// convention (raw pointers vs. memref_t struct, see ukernel.h vs.
// ukernel.cpp from the GEMM path) is unverified for this new kernel and
// may need adjusting once you see how xchesscc actually lowers the
// external_func call in the generated MLIR.

#include <stdint.h>

#define FEATURE_COUNT 840

static inline float bf16_to_f32(uint16_t x) {
    uint32_t y = (uint32_t)x << 16;
    return *((float*)&y);
}

extern "C" {

void zero_score(float* out) {
    out[0] = 0.0f;
}

void dotproduct_bf16_f32(uint16_t* features, uint16_t* weights, float* out) {
    float acc = 0.0f;
    for (int i = 0; i < FEATURE_COUNT; i++) {
        acc += bf16_to_f32(features[i]) * bf16_to_f32(weights[i]);
    }
    out[0] = acc;
}

}