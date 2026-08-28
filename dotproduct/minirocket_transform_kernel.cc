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
// x_full:    pointer to the FULL combined input buffer (COMBINED_IN_LEN
//            elements: [series | bias_slice | smean_slice | sscale_slice]),
//            NOT a pre-sliced SEQ_LENGTH-only buffer. This function reads
//            the series starting at x_full[series_offset] internally.
// series_offset: offset of the series data within x_full (0, per the
//            generator's SERIES_OFFSET convention).
// out_full:  pointer to the FULL per-tile output buffer
//            (ACTIVE_FEATURES_PER_TILE elements), NOT a 1-element scratch
//            buffer. This function writes its result to
//            out_full[out_index] internally.
// out_index: which slot in out_full this call's result belongs to.
//
// CHANGED from the original scratch-buffer design (see FUSION_PLAN.md /
// commit history): the original version took a pre-copied SEQ_LENGTH
// series pointer and a 1-element output scratch pointer, requiring the
// MLIR generator to memref.alloc() both a series_local copy buffer and
// an out_scratch buffer. memref.alloc() lowers to a REAL HEAP ALLOCATION
// requiring libc's malloc pool (_ZL4pool) to be linked in -- and on this
// hardware's AIE-ML tile, all 4 local memory banks (dm_bank0-3, 16KB
// each = 64KB total) are already fully claimed by the toolchain's
// default static allocations, leaving ZERO free space for a heap pool.
// Confirmed by comparing against the proven, hardware-validated
// classifier kernel (dotproduct12_v3.mlir/.cc): its map files contain NO
// _ZL4pool/malloc/pool symbols at all, because it never calls
// memref.alloc() -- it only reads/writes ObjectFifo-provided buffers
// directly. This version does the same: no local scratch buffers, no
// heap allocation, direct indexing into the full input/output buffers
// the caller already has.
void minirocket_feature_transform(
    int32_t* x_full,
    int32_t series_offset,
    int32_t bias,
    int32_t scaler_mean,
    int32_t scaler_scale,
    int32_t dilation,
    int32_t padding,
    int32_t kernel_i0,
    int32_t kernel_i1,
    int32_t kernel_i2,
    int32_t padding1,
    int32_t* out_full,
    int32_t out_index
) {
    // A = -X, G = 3X (x_full[series_offset + t] replaces the old x[t])
    int32_t A[SEQ_LENGTH];
    int32_t G[SEQ_LENGTH];
    for (int t = 0; t < SEQ_LENGTH; t++) {
        int32_t xv = x_full[series_offset + t];
        A[t] = -xv;
        G[t] = xv + xv + xv;
    }

    // C = C_alpha + C_gamma[i0] + C_gamma[i1] + C_gamma[i2].
    //
    // MEMORY FIX (see FUSION_PLAN.md / handoff notes): the original version
    // of this function stored the FULL C_gamma[KERNEL_TAPS][SEQ_LENGTH]
    // table (9 x 251 x 4 bytes = ~9KB), mirroring minirocket.py's CPU
    // reference structure -- but that CPU version reuses the full table
    // across all 84 kernel-index-triples for one dilation, while THIS
    // function only ever needs 3 of the 9 rows (kernel_i0/i1/i2) since it
    // recomputes from scratch per feature (Path A, no cross-feature
    // caching). Storing all 9 rows here was pure waste and caused a real
    // linker failure on hardware: "could not find free space for
    // SpaceSymbol _ZL4pool in memory DMb" -- an AIE-ML compute tile has
    // only 64KB of local data memory total (confirmed: AMD AIE-ML
    // architecture, 4x16KB banks), and combined with per-call stack usage
    // this exceeded what one bank could hold once the libc heap pool
    // symbol also needed space.
    //
    // Fix: instead of storing a 9-row table, build C directly -- for each
    // of the KERNEL_TAPS-1 shift-accumulate passes, check whether the
    // CURRENT gamma_index matches any of kernel_i0/i1/i2, and if so, add
    // that pass's G-contribution directly into a per-position running sum
    // (init'd from C_alpha), rather than storing it into a persistent
    // 251-wide row that's kept around for the whole function. This drops
    // per-call stack usage from ~13KB (9 C_gamma rows + A/G/C_alpha/C) to
    // ~4KB (A/G/C_alpha/C only, no C_gamma table at all).
    int32_t C[SEQ_LENGTH];   // starts as C_alpha, becomes C_alpha + selected C_gamma rows
    for (int t = 0; t < SEQ_LENGTH; t++) {
        C[t] = A[t];
    }

    // gamma_index == KERNEL_TAPS/2 (the "center" pass) is G itself,
    // un-shifted -- matches minirocket.py's C_gamma[9//2] = G initializer.
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
            C[dst_idx] += A[t];  // C_alpha's own accumulation, always happens
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
            C[t] += A[start + t];  // C_alpha's own accumulation, always happens
            if (needed) {
                C[t] += G[start + t];
            }
        }
        start += dilation;
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

    out_full[out_index] = norm;
}

}