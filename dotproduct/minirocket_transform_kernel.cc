// minirocket_transform_kernel.cc
//
// DATA-DRIVEN VERSION. See minirocket_transform_kernel.py for the
// matching MLIR generator. This replaces the per-feature-call-site
// version that hit "Overflow of program memory" at 420 features/tile.
//
// minirocket_transform_tile() processes an entire tile's features in
// ONE call, looping internally over a parameter array delivered as
// data. Per-feature math (compute_one_feature) is UNCHANGED from the
// version already verified correct against real NPU hardware output
// (16/16 exact bit-for-bit matches at test scale).

#include <stdint.h>

#define SEQ_LENGTH 251
#define KERNEL_TAPS 9
#define SCALE 1000
#define PARAMS_PER_FEATURE 6

extern "C" {

void zero_i32_scalar(int32_t* out) {
    out[0] = 0;
}

static int32_t compute_one_feature(
    int32_t* x,
    int32_t bias,
    int32_t scaler_mean,
    int32_t scaler_scale,
    int32_t dilation,
    int32_t padding,
    int32_t kernel_i0,
    int32_t kernel_i1,
    int32_t kernel_i2,
    int32_t padding1
) {
    int32_t A[SEQ_LENGTH];
    int32_t G[SEQ_LENGTH];
    for (int t = 0; t < SEQ_LENGTH; t++) {
        A[t] = -x[t];
        G[t] = x[t] + x[t] + x[t];
    }

    int32_t C[SEQ_LENGTH];
    for (int t = 0; t < SEQ_LENGTH; t++) {
        C[t] = A[t];
    }

    int center = KERNEL_TAPS / 2;
    if (center == kernel_i0 || center == kernel_i1 || center == kernel_i2) {
        for (int t = 0; t < SEQ_LENGTH; t++) {
            C[t] += G[t];
        }
    }

    int32_t end = SEQ_LENGTH - padding;
    for (int gamma_index = 0; gamma_index < center; gamma_index++) {
        bool needed = (gamma_index == kernel_i0) || (gamma_index == kernel_i1) || (gamma_index == kernel_i2);
        for (int t = 0; t < end; t++) {
            int32_t dst_idx = (SEQ_LENGTH - end) + t;
            C[dst_idx] += A[t];
            if (needed) {
                C[dst_idx] += G[t];
            }
        }
        end += dilation;
    }

    int32_t start = dilation;
    for (int gamma_index = center + 1; gamma_index < KERNEL_TAPS; gamma_index++) {
        bool needed = (gamma_index == kernel_i0) || (gamma_index == kernel_i1) || (gamma_index == kernel_i2);
        for (int t = 0; t < SEQ_LENGTH - start; t++) {
            C[t] += A[start + t];
            if (needed) {
                C[t] += G[start + t];
            }
        }
        start += dilation;
    }

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

    int32_t eps = 1;
    int32_t diff = ppv_scaled - scaler_mean;
    int32_t denom = scaler_scale + eps;
    int32_t norm = (diff * SCALE) / denom;

    int32_t clip_hi = 3 * SCALE;
    int32_t clip_lo = -3 * SCALE;
    if (norm > clip_hi) norm = clip_hi;
    if (norm < clip_lo) norm = clip_lo;

    return norm;
}

void minirocket_transform_tile(
    int32_t* x,
    int32_t* bias,
    int32_t* scaler_mean,
    int32_t* scaler_scale,
    int32_t* params,
    int32_t num_features,
    int32_t* out
) {
    for (int32_t f = 0; f < num_features; f++) {
        int32_t dilation   = params[f * PARAMS_PER_FEATURE + 0];
        int32_t padding    = params[f * PARAMS_PER_FEATURE + 1];
        int32_t kernel_i0  = params[f * PARAMS_PER_FEATURE + 2];
        int32_t kernel_i1  = params[f * PARAMS_PER_FEATURE + 3];
        int32_t kernel_i2  = params[f * PARAMS_PER_FEATURE + 4];
        int32_t padding1   = params[f * PARAMS_PER_FEATURE + 5];

        out[f] = compute_one_feature(
            x, bias[f], scaler_mean[f], scaler_scale[f],
            dilation, padding, kernel_i0, kernel_i1, kernel_i2, padding1
        );
    }
}

}