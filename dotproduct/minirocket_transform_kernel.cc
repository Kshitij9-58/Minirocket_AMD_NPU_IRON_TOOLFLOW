// minirocket_transform_kernel.cc
//
// Real MiniROCKET transform compute function: for ONE feature (one
// dilation + one kernel-index-triple), computes the dilated convolution
// C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2], then PPV
// (mean(C > bias), full-range or [padding:-padding] trimmed depending
// on padding1), then normalizes: (ppv - scaler_mean) / (scaler_scale + eps),
// clipped to [-3, 3].
//
// Calling convention matches the project's proven pattern (see
// dotproduct12_v3.mlir's dotproduct_partial_bf16_f32 signature): extern
// "C", raw pointer args, no memref wrapper types.
//
// Fixed-point: int32 throughout, values scaled by SCALE=1000, matching
// the existing transform kernel's convention (static_biases_int =
// (real_biases * 1000).astype(np.int32) in npu_end_to_end.py /
// dotproduct_minirocket.py). Host must deliver bias, scaler_mean,
// scaler_scale pre-scaled by the same factor.
//
// Per-feature dilation/kernel-index/padding values are baked in as
// COMPILE-TIME constants by the MLIR generator (minirocket_transform_kernel.py)
// -- one specialized function call per feature, per the confirmed 840-entry
// feature schedule (build_schedule.py). This file provides the GENERIC
// implementation parameterized by those values; the generator calls this
// same function once per feature with different dilation/padding/index
// arguments, matching how dotproduct_partial_bf16_f32 is called once per
// tile with the same function but different data each time.
//
// STATUS: new for Step 1 of the on-chip-execution fusion work. Not yet
// compiled with xchesscc, not yet run on hardware. Write, review, then
// verify build + Step 3's CPU-comparison gate before trusting output.

#include <stdint.h>

#define SEQ_LENGTH 251
#define KERNEL_TAPS 9
#define SCALE 1000

extern "C" {

// Zero-initializes a single int32 output slot. Mirrors zero_f32_scalar's
// role in the classifier kernel.
void zero_i32_scalar(int32_t* out) {
    out[0] = 0;
}

// Computes ONE feature's full transform: dilated conv -> PPV -> normalize.
//
// x:            SEQ_LENGTH int32 values, the raw time series (scaled by
//               SCALE), shared across all features in this tile -- same
//               buffer passed on every call, only dilation/kernel_index/
//               padding/bias/scaler args change per call.
// bias:         single int32 value (scaled by SCALE), this feature's bias.
// scaler_mean:  single int32 value (scaled by SCALE), this feature's
//               scaler mean.
// scaler_scale: single int32 value (scaled by SCALE), this feature's
//               scaler scale.
// dilation:     this feature's dilation (compile-time constant per call
//               site, passed as a runtime int here for a single generic
//               function -- see minirocket_transform_kernel.py for the
//               per-feature call-site generation).
// padding:      ((KERNEL_TAPS-1)*dilation)/2, precomputed by the caller.
// kernel_i0/i1/i2: this feature's kernel-index-triple (0..8 each).
// padding1:     0 = full-range PPV, 1 = [padding:-padding] trimmed PPV.
// out:          single int32 output slot, this feature's normalized
//               value. NOT accumulated (unlike the classifier's
//               dotproduct_partial, this is a single independent
//               feature computation, not a partial sum) -- overwrites,
//               matching the zero_i32_scalar + single-call pattern per
//               feature.
void minirocket_feature_transform(
    int32_t* x,
    int32_t bias,
    int32_t scaler_mean,
    int32_t scaler_scale,
    int32_t dilation,
    int32_t padding,
    int32_t kernel_i0,
    int32_t kernel_i1,
    int32_t kernel_i2,
    int32_t padding1,
    int32_t* out
) {
    // A = -X, G = 3X
    int32_t A[SEQ_LENGTH];
    int32_t G[SEQ_LENGTH];
    for (int t = 0; t < SEQ_LENGTH; t++) {
        A[t] = -x[t];
        G[t] = x[t] + x[t] + x[t];
    }

    // C_alpha starts as A; C_gamma[KERNEL_TAPS/2] (center) starts as G;
    // both built via KERNEL_TAPS-1 shifted accumulation passes, matching
    // minirocket.py's transform() exactly.
    int32_t C_alpha[SEQ_LENGTH];
    int32_t C_gamma[KERNEL_TAPS][SEQ_LENGTH];

    for (int t = 0; t < SEQ_LENGTH; t++) {
        C_alpha[t] = A[t];
    }
    for (int t = 0; t < SEQ_LENGTH; t++) {
        C_gamma[KERNEL_TAPS / 2][t] = G[t];
    }

    int32_t end = SEQ_LENGTH - padding;
    for (int gamma_index = 0; gamma_index < KERNEL_TAPS / 2; gamma_index++) {
        for (int t = 0; t < end; t++) {
            int32_t dst_idx = (SEQ_LENGTH - end) + t;
            C_alpha[dst_idx] += A[t];
            C_gamma[gamma_index][dst_idx] = G[t];
        }
        end += dilation;
    }

    int32_t start = dilation;
    for (int gamma_index = KERNEL_TAPS / 2 + 1; gamma_index < KERNEL_TAPS; gamma_index++) {
        for (int t = 0; t < SEQ_LENGTH - start; t++) {
            C_alpha[t] += A[start + t];
            C_gamma[gamma_index][t] = G[start + t];
        }
        start += dilation;
    }

    // C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2]
    int32_t C[SEQ_LENGTH];
    for (int t = 0; t < SEQ_LENGTH; t++) {
        C[t] = C_alpha[t] + C_gamma[kernel_i0][t] + C_gamma[kernel_i1][t] + C_gamma[kernel_i2][t];
    }

    // PPV: mean(C > bias), full range or [padding:-padding] trimmed,
    // depending on padding1. Result scaled by SCALE to stay in the
    // fixed-point convention (ppv_scaled = count * SCALE / n_samples).
    int32_t range_start = (padding1 == 0) ? 0 : padding;
    int32_t range_end = (padding1 == 0) ? SEQ_LENGTH : (SEQ_LENGTH - padding);
    int32_t n_samples = range_end - range_start;

    int32_t count = 0;
    for (int t = range_start; t < range_end; t++) {
        if (C[t] > bias) {
            count += 1;
        }
    }
    int32_t ppv_scaled = (count * SCALE) / n_samples;

    // Normalize: (ppv - scaler_mean) / (scaler_scale + eps), clipped to
    // [-3*SCALE, 3*SCALE].
    int32_t eps = 1;
    int32_t diff = ppv_scaled - scaler_mean;
    int32_t denom = scaler_scale + eps;
    int32_t norm = (diff * SCALE) / denom;

    int32_t clip_hi = 3 * SCALE;
    int32_t clip_lo = -3 * SCALE;
    if (norm > clip_hi) norm = clip_hi;
    if (norm < clip_lo) norm = clip_lo;

    out[0] = norm;
}

}