#!/usr/bin/env python3
"""
prep_and_verify_fused_pipeline.py

Builds the two host input files the fused 14-tile pipeline needs:
  1. combined_transform_input.txt -- 8062 int32 values (both transform
     tiles' full input: series+bias+scaler_mean+scaler_scale+params)
  2. classifier_weights.txt -- 840 float values (Ridge classifier
     coefficients for ONE class, matching the classifier's own proven
     840-length weight vector)

Also computes the END-TO-END CPU reference score (real MiniROCKET
transform -> real dot product against the given weights) for comparison
against the fused pipeline's hardware output.

Usage:
  python3 prep_and_verify_fused_pipeline.py
  python3 prep_and_verify_fused_pipeline.py --compare fused_pipeline_output.txt
"""

import argparse
import json
from itertools import combinations

import numpy as np

SEQ_LENGTH = 251
DILATIONS = [1, 2, 3, 4, 6, 9, 14, 21, 31]
NUM_FEATURES_PER_DILATION = [2, 1, 1, 1, 1, 1, 1, 1, 1]
NUM_KERNEL_INDICES = 84
KERNEL_TAPS = 9
FEATURES_PER_TILE = 420
TOTAL_FEATURES = 840
PARAMS_PER_FEATURE = 6
SCALE = 1000
CLIP_VAL = 3.0

KERNEL_INDICES = list(combinations(range(KERNEL_TAPS), 3))


def build_feature_schedule():
    schedule = []
    for dilation_index, dilation in enumerate(DILATIONS):
        padding0 = dilation_index % 2
        n_feat = NUM_FEATURES_PER_DILATION[dilation_index]
        padding = ((KERNEL_TAPS - 1) * dilation) // 2
        for kernel_index in range(NUM_KERNEL_INDICES):
            padding1 = (padding0 + kernel_index) % 2
            i0, i1, i2 = KERNEL_INDICES[kernel_index]
            for _ in range(n_feat):
                schedule.append({
                    "dilation": dilation, "padding": padding,
                    "kernel_i0": i0, "kernel_i1": i1, "kernel_i2": i2,
                    "padding1": padding1,
                })
    assert len(schedule) == TOTAL_FEATURES
    return schedule


def cpu_reference_feature_fixedpoint(x_scaled, bias_scaled, dilation, padding,
                                      i0, i1, i2, padding1):
    """Matches minirocket_feature_transform's exact fixed-point arithmetic."""
    x = x_scaled.astype(np.int64)
    A = -x
    G = x + x + x
    C = A.copy()
    center = KERNEL_TAPS // 2
    if center in (i0, i1, i2):
        C = C + G
    end = SEQ_LENGTH - padding
    for gamma_index in range(center):
        needed = gamma_index in (i0, i1, i2)
        dst = slice(SEQ_LENGTH - end, SEQ_LENGTH)
        C[dst] += A[:end]
        if needed:
            C[dst] += G[:end]
        end += dilation
    start = dilation
    for gamma_index in range(center + 1, KERNEL_TAPS):
        needed = gamma_index in (i0, i1, i2)
        C[:SEQ_LENGTH - start] += A[start:]
        if needed:
            C[:SEQ_LENGTH - start] += G[start:]
        start += dilation

    if padding1 == 0:
        range_start, range_end = 0, SEQ_LENGTH
    else:
        range_start, range_end = padding, SEQ_LENGTH - padding
    n_samples = range_end - range_start
    count = int(np.sum(C[range_start:range_end] > bias_scaled))
    ppv_scaled = (count * SCALE) // n_samples
    return ppv_scaled


import struct


def float_to_bf16(f):
    """Same bit-truncation as the C++ float_to_bf16 / float_to_bf16_trunc."""
    bits = struct.unpack('<I', struct.pack('<f', float(f)))[0]
    return (bits >> 16) & 0xFFFF


def bf16_to_float(bf16_val):
    bits = (int(bf16_val) & 0xFFFF) << 16
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minirocket_model.json")
    ap.add_argument("--test-data", default="minirocket_model_test_data.json")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--transform-out", default="combined_transform_input.txt")
    ap.add_argument("--weights-out", default="classifier_weights.txt")
    ap.add_argument("--compare", default=None)
    args = ap.parse_args()

    with open(args.model) as f:
        model = json.load(f)
    with open(args.test_data) as f:
        test_data = json.load(f)

    biases = np.array(model["biases"], dtype=np.float64)
    scaler_mean = np.array(model["scaler_mean"], dtype=np.float64)
    scaler_scale = np.array(model["scaler_scale"], dtype=np.float64)
    X_test = np.array(test_data["X_test"], dtype=np.float64)
    x_sample = X_test[args.sample_index]
    assert len(x_sample) == SEQ_LENGTH

    # Classifier weights: use the model's real trained Ridge coefficients
    # for class 0 (matching the existing classifier driver's convention
    # of testing against one class's weight vector at a time)
    classifier_coef = np.array(model["classifier_coef"], dtype=np.float64)
    weights = classifier_coef[0]  # class 0's 840 coefficients
    assert len(weights) == TOTAL_FEATURES, f"expected {TOTAL_FEATURES} weights, got {len(weights)}"

    schedule = build_feature_schedule()
    x_scaled = np.round(x_sample * SCALE).astype(np.int64)

    # --- Build transform combined input (both tiles) ---
    print("Building transform combined input (840 features across 2 tiles)...")
    combined = []
    for t in range(2):
        base = t * FEATURES_PER_TILE
        tile_schedule = schedule[base:base + FEATURES_PER_TILE]
        tile_bias = biases[base:base + FEATURES_PER_TILE]
        tile_smean = scaler_mean[base:base + FEATURES_PER_TILE]
        tile_sscale = scaler_scale[base:base + FEATURES_PER_TILE]

        combined.extend(x_scaled.tolist())
        combined.extend(np.round(tile_bias * SCALE).astype(np.int64).tolist())
        combined.extend(np.round(tile_smean * SCALE).astype(np.int64).tolist())
        combined.extend(np.round(tile_sscale * SCALE).astype(np.int64).tolist())
        for entry in tile_schedule:
            combined.extend([
                entry["dilation"], entry["padding"],
                entry["kernel_i0"], entry["kernel_i1"], entry["kernel_i2"],
                entry["padding1"],
            ])

    expected_len = 2 * (SEQ_LENGTH + 3 * FEATURES_PER_TILE + FEATURES_PER_TILE * PARAMS_PER_FEATURE)
    assert len(combined) == expected_len, f"expected {expected_len}, got {len(combined)}"

    with open(args.transform_out, "w") as f:
        for v in combined:
            f.write(f"{v}\n")
    print(f"Wrote {len(combined)} values to {args.transform_out}")

    # --- Write classifier weights (plain floats, runner converts to bf16) ---
    with open(args.weights_out, "w") as f:
        for v in weights:
            f.write(f"{v}\n")
    print(f"Wrote {len(weights)} weight values to {args.weights_out}")

    # --- Compute END-TO-END CPU reference: real transform -> real dot product ---
    print("\nComputing end-to-end CPU reference (transform + dot product)...")
    transform_features_norm = []
    for feat_idx, entry in enumerate(schedule):
        bias_scaled = int(round(biases[feat_idx] * SCALE))
        smean_scaled = int(round(scaler_mean[feat_idx] * SCALE))
        sscale_scaled = int(round(scaler_scale[feat_idx] * SCALE))

        ppv_scaled = cpu_reference_feature_fixedpoint(
            x_scaled, bias_scaled, entry["dilation"], entry["padding"],
            entry["kernel_i0"], entry["kernel_i1"], entry["kernel_i2"],
            entry["padding1"]
        )
        diff = ppv_scaled - smean_scaled
        denom = sscale_scaled + 1
        norm_scaled = int((diff * SCALE) / denom)
        clip_hi = int(CLIP_VAL * SCALE)
        norm_scaled = max(-clip_hi, min(clip_hi, norm_scaled))
        # Convert through bf16 truncation, same as the on-chip kernel does
        norm_real = norm_scaled / SCALE
        norm_bf16 = float_to_bf16(norm_real)
        norm_bf16_real = float(bf16_to_float(norm_bf16))
        transform_features_norm.append(norm_bf16_real)

    transform_features_norm = np.array(transform_features_norm, dtype=np.float64)

    # Dot product, weights also truncated through bf16 (matching on-chip precision)
    weights_bf16 = np.array([float(bf16_to_float(float_to_bf16(w))) for w in weights])
    per_tile_partials = []
    for col in range(3):
        col_start = col * 280
        for sub in range(4):
            chunk_start = col_start + sub * 70
            chunk_end = chunk_start + 70
            partial = float(np.sum(
                transform_features_norm[chunk_start:chunk_end] * weights_bf16[chunk_start:chunk_end]
            ))
            per_tile_partials.append(partial)

    final_score = sum(per_tile_partials)
    print("\n=== CPU REFERENCE (end-to-end: real transform + real dot product) ===")
    for i, p in enumerate(per_tile_partials):
        print(f"  tile {i}: {p:.6f}")
    print(f"\nReference final score: {final_score:.6f}")

    with open("cpu_reference_fused_pipeline.txt", "w") as f:
        for p in per_tile_partials:
            f.write(f"{p}\n")

    if args.compare:
        npu_vals = []
        with open(args.compare) as f:
            for line in f:
                line = line.strip()
                if line:
                    npu_vals.append(float(line))
        print(f"\n=== COMPARISON ===")
        npu_final = sum(npu_vals)
        print(f"NPU final score: {npu_final:.6f}")
        print(f"CPU reference final score: {final_score:.6f}")
        print(f"Absolute diff: {abs(npu_final - final_score):.6f}")
        for i in range(min(len(npu_vals), len(per_tile_partials))):
            diff = abs(npu_vals[i] - per_tile_partials[i])
            status = "OK" if diff < 0.01 else "MISMATCH"
            print(f"  tile {i}: cpu={per_tile_partials[i]:.6f} npu={npu_vals[i]:.6f} diff={diff:.6f} [{status}]")


if __name__ == "__main__":
    main()