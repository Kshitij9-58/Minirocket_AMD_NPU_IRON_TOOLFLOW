#!/usr/bin/env python3
"""
prep_and_verify_test8.py

Two jobs:
  1. Build combined_input.txt -- the input buffer the transform kernel
     expects, packed in its exact layout, from real test data.
  2. Compute the CPU reference values for the same 16 features, so the
     NPU output can be compared against ground truth.

Run BEFORE transform_test8_runner (to make combined_input.txt), then
again with --compare after (to check the NPU's output).

Usage:
  python3 prep_and_verify_test8.py                       # generate input + print reference
  python3 prep_and_verify_test8.py --compare npu_out.txt # compare NPU output to reference
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
FEATURES_PER_TILE_FULL = 420
TEST_FEATURE_LIMIT = 8
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
    assert len(schedule) == 840, f"expected 840, got {len(schedule)}"
    return schedule


def cpu_reference_feature_fixedpoint(x_scaled, bias_scaled, dilation, padding,
                                      i0, i1, i2, padding1):
    """Mirrors minirocket_transform_kernel.cc EXACTLY, including its
    integer arithmetic, so the comparison is apples-to-apples with what
    the NPU actually computes (not the float reference, which would
    differ slightly due to fixed-point rounding)."""
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="minirocket_model.json")
    ap.add_argument("--test-data", default="minirocket_model_test_data.json")
    ap.add_argument("--sample-index", type=int, default=0)
    ap.add_argument("--out", default="combined_input.txt")
    ap.add_argument("--compare", default=None,
                     help="Path to file with NPU output (16 int values), to compare against reference")
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

    schedule = build_feature_schedule()
    tiles = []
    for t in range(2):
        base = t * FEATURES_PER_TILE_FULL
        tiles.append({
            "schedule": schedule[base:base + FEATURES_PER_TILE_FULL][:TEST_FEATURE_LIMIT],
            "bias": biases[base:base + FEATURES_PER_TILE_FULL][:TEST_FEATURE_LIMIT],
            "smean": scaler_mean[base:base + FEATURES_PER_TILE_FULL][:TEST_FEATURE_LIMIT],
            "sscale": scaler_scale[base:base + FEATURES_PER_TILE_FULL][:TEST_FEATURE_LIMIT],
        })

    x_scaled = np.round(x_sample * SCALE).astype(np.int64)

    # --- Build combined input buffer ---
    combined = []
    for t in range(2):
        combined.extend(x_scaled.tolist())
        combined.extend(np.round(tiles[t]["bias"] * SCALE).astype(np.int64).tolist())
        combined.extend(np.round(tiles[t]["smean"] * SCALE).astype(np.int64).tolist())
        combined.extend(np.round(tiles[t]["sscale"] * SCALE).astype(np.int64).tolist())

    expected_len = 2 * (SEQ_LENGTH + 3 * TEST_FEATURE_LIMIT)
    assert len(combined) == expected_len, f"expected {expected_len}, got {len(combined)}"

    with open(args.out, "w") as f:
        for v in combined:
            f.write(f"{v}\n")
    print(f"Wrote {len(combined)} values to {args.out}")

    # --- CPU reference ---
    print("\n=== CPU REFERENCE (fixed-point, matches kernel arithmetic) ===")
    reference = []
    for t in range(2):
        for feat_idx, entry in enumerate(tiles[t]["schedule"]):
            bias_scaled = int(round(tiles[t]["bias"][feat_idx] * SCALE))
            smean_scaled = int(round(tiles[t]["smean"][feat_idx] * SCALE))
            sscale_scaled = int(round(tiles[t]["sscale"][feat_idx] * SCALE))

            ppv_scaled = cpu_reference_feature_fixedpoint(
                x_scaled, bias_scaled, entry["dilation"], entry["padding"],
                entry["kernel_i0"], entry["kernel_i1"], entry["kernel_i2"],
                entry["padding1"]
            )
            diff = ppv_scaled - smean_scaled
            denom = sscale_scaled + 1
            norm = int((diff * SCALE) / denom) if denom != 0 else 0
            clip_hi = int(CLIP_VAL * SCALE)
            norm = max(-clip_hi, min(clip_hi, norm))
            reference.append(norm)
            print(f"  tile{t} feat{feat_idx}: dilation={entry['dilation']:3d} "
                  f"pad1={entry['padding1']} ppv_scaled={ppv_scaled:5d} norm={norm}")

    if args.compare:
        npu_vals = []
        with open(args.compare) as f:
            for line in f:
                line = line.strip()
                if line:
                    npu_vals.append(int(line))
        print(f"\n=== COMPARISON (loaded {len(npu_vals)} NPU values) ===")
        if len(npu_vals) != len(reference):
            print(f"WARNING: count mismatch -- reference has {len(reference)}, NPU file has {len(npu_vals)}")
        n = min(len(npu_vals), len(reference))
        mismatches = 0
        for i in range(n):
            match = (npu_vals[i] == reference[i])
            if not match:
                mismatches += 1
            status = "OK" if match else "MISMATCH"
            print(f"  [{i:2d}] reference={reference[i]:6d}  npu={npu_vals[i]:6d}  [{status}]")
        print(f"\n{n - mismatches}/{n} exact matches")
        print("PASS" if mismatches == 0 else f"FAIL ({mismatches} mismatches)")


if __name__ == "__main__":
    main()
