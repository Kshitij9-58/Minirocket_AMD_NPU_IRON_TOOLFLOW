#include <stdint.h>

extern "C" {

// Helper to convert BF16 bits to Float32 for scalar math
static inline float bf16_to_float(uint16_t x) {
    uint32_t y = (uint32_t)x << 16;
    return *((float*)&y);
}

// Zero out the entire 64x32 output buffer (Matching your MLIR shape)
void zero_f32(float* c) {
    for(int i = 0; i < 64 * 32; i++) {
        c[i] = 0.0f;
    }
}

// 64x64 by 64x32 MatMul Kernel 
// Optimized for typical MiniRocket tiling sizes
void matmul_bf16_f32(uint16_t* a, uint16_t* b, float* c) {
    for(int i = 0; i < 64; i++) {
        for(int j = 0; j < 32; j++) {
            float sum = 0.0f;
            for(int k = 0; k < 64; k++) {
                // a: [64 x 64], b: [64 x 32], c: [64 x 32]
                sum += bf16_to_float(a[i * 64 + k]) * bf16_to_float(b[k * 32 + j]);
            }
            c[i * 32 + j] += sum;
        }
    }
}

}