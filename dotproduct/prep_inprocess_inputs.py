#!/usr/bin/env python3
"""
prep_inprocess_inputs.py

Builds the files pipeline_inprocess_runner needs, from minirocket_model.json
and minirocket_model_test_data.json (run from the dotproduct directory):

  batch_series.txt      one line per test sample, 251 ints (series * 1000)
  batch_base_input.txt  8062 ints: the two transform tiles' combined input
                        (series slot holds sample 0; the runner overwrites it
                        per sample, everything else is constant across samples)
  weights_0.txt ...     per-class weights from model['classifier_coef'],
                        written with fmt '%.6f', the same call the validated
                        driver dotproduct_minirocket.py makes
"""

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
                schedule.append((dilation, padding, i0, i1, i2, padding1))
    assert len(schedule) == TOTAL_FEATURES
    return schedule


def main():
    with open("minirocket_model.json") as f:
        model = json.load(f)
    with open("minirocket_model_test_data.json") as f:
        test = json.load(f)

    X = np.array(test["X_test"], dtype=np.float64)
    assert X.shape[1] == SEQ_LENGTH, f"expected series length {SEQ_LENGTH}, got {X.shape[1]}"
    n = X.shape[0]

    series_scaled = np.round(X * SCALE).astype(np.int64)
    with open("batch_series.txt", "w") as f:
        for row in series_scaled:
            f.write(" ".join(str(int(v)) for v in row) + "\n")

    biases = np.array(model["biases"], dtype=np.float64)
    smean = np.array(model["scaler_mean"], dtype=np.float64)
    sscale = np.array(model["scaler_scale"], dtype=np.float64)
    schedule = build_feature_schedule()

    base = []
    for t in range(2):
        lo = t * FEATURES_PER_TILE
        hi = lo + FEATURES_PER_TILE
        base.extend(series_scaled[0].tolist())
        base.extend(np.round(biases[lo:hi] * SCALE).astype(np.int64).tolist())
        base.extend(np.round(smean[lo:hi] * SCALE).astype(np.int64).tolist())
        base.extend(np.round(sscale[lo:hi] * SCALE).astype(np.int64).tolist())
        for entry in schedule[lo:hi]:
            base.extend(entry)
    expected = 2 * (SEQ_LENGTH + 3 * FEATURES_PER_TILE + FEATURES_PER_TILE * PARAMS_PER_FEATURE)
    assert len(base) == expected, f"expected {expected}, got {len(base)}"
    with open("batch_base_input.txt", "w") as f:
        for v in base:
            f.write(f"{int(v)}\n")

    weights = np.array(model["classifier_coef"], dtype=np.float32)
    assert weights.shape == (3, TOTAL_FEATURES), f"unexpected weights shape {weights.shape}"
    for c in range(weights.shape[0]):
        np.savetxt(f"weights_{c}.txt", weights[c], fmt="%.6f")

    print(f"samples: {n}")
    print(f"wrote batch_series.txt ({n} lines x {SEQ_LENGTH})")
    print(f"wrote batch_base_input.txt ({len(base)} values)")
    print(f"wrote weights_0.txt .. weights_{weights.shape[0] - 1}.txt ({TOTAL_FEATURES} values each)")


if __name__ == "__main__":
    main()
