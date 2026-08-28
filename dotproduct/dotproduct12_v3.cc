// dotproduct12_v3.cc
//
// STATUS: DRAFT, UNBUILT, UNVERIFIED. Not compiled with xchesscc, not run
// on hardware.
//
// Direct structural mirror of the real, working GEMM classifier's
// microkernel (matmul_bf16_f32 / zero_f32, from final-12.xclbin's source
// MLIR). Same two-function shape (a "zero" initializer + one "compute"
// function called in a loop), same bf16-in/f32-out convention. Only the
// math changes: a 70-element partial dot product instead of a 64x64
// padded matmul.
//
// Each of the 12 compute tiles (3 columns x 4 tiles, matching GEMM's
// real topology exactly) gets its own 70-element chunk of the 840-wide
// feature/weight vectors (840 / 12 = 70, clean split, no padding needed
// -- unlike GEMM's 64x64 blocks, which pad a real ~10-70 element chunk
// up to 4096 elements, ~98% waste per the project's own Limitations
// section).

#include <stdint.h>

#define CHUNK_SIZE 70   // 840 / 12 compute tiles, no padding required

static inline float bf16_to_f32(uint16_t x) {
    uint32_t y = (uint32_t)x << 16;
    return *((float*)&y);
}

extern "C" {

// Mirrors zero_f32(memref<64x32xf32>) -- same role (initialize the
// accumulator before the compute loop), scalar output instead of a
// 64x32 block since a dot-product partial is a single f32, not a matrix.
void zero_f32_scalar(float* out) {
    out[0] = 0.0f;
}

// Mirrors matmul_bf16_f32(memref<64x64xbf16>, memref<64x32xbf16>, memref<64x32xf32>).
// Computes a partial dot product over this tile's 70-element chunk and
// ACCUMULATES into *out (matches GEMM's pattern of calling matmul inside
// an 8-iteration inner loop, each call adding into the same accumulator
// buffer -- here there's no inner loop needed since one call covers the
// tile's whole chunk, but the accumulate-into-out semantics match).
void dotproduct_partial_bf16_f32(uint16_t* features_chunk, uint16_t* weights_chunk, float* out) {
    float acc = 0.0f;
    for (int i = 0; i < CHUNK_SIZE; i++) {
        acc += bf16_to_f32(features_chunk[i]) * bf16_to_f32(weights_chunk[i]);
    }
    out[0] += acc;
}

}
