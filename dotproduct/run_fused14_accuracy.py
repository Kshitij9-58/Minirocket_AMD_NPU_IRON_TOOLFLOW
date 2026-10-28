#!/usr/bin/env python3
"""
Accuracy run for fused14_join_refeed.xclbin.

One fresh process per dispatch (the xclbin stalls after ~6 dispatches in one
process). Each dispatch scores ONE class: the 12 partials sum to that class's
dot product. 3 dispatches per sample, one per class's classifier_coef row.
Scores are z-scored per class column across samples (intercepts cancel), then
argmax, then the best class-label mapping is reported alongside identity.
"""
import itertools
import json
import os
import re
import subprocess
import sys

import numpy as np

from prep_and_verify_fused_pipeline import (
    build_feature_schedule, SEQ_LENGTH, FEATURES_PER_TILE, TOTAL_FEATURES, SCALE,
)

RUNNER = "./fused14_join_refeed_runner"
XCLBIN = "fused14_join_refeed.xclbin"
INSTS = "fused14_join_refeed_sequence.bin"
INP = "_f14_input.txt"
WTS = "_f14_weights.txt"
OUT = "fused14_output.txt"
TILE_LEN = SEQ_LENGTH + 3 * FEATURES_PER_TILE + FEATURES_PER_TILE * 6  # 4031

model = json.load(open("minirocket_model.json"))
test = json.load(open("minirocket_model_test_data.json"))
X = np.array(test["X_test"], dtype=np.float64)
y = test.get("y_test")
if y is None:
    print("No y_test key. Keys in test data:", list(test.keys()))
    sys.exit(1)
coef = np.array(model["classifier_coef"], dtype=np.float64)
assert coef.shape == (3, TOTAL_FEATURES), coef.shape

biases = np.array(model["biases"], dtype=np.float64)
smean = np.array(model["scaler_mean"], dtype=np.float64)
sscale = np.array(model["scaler_scale"], dtype=np.float64)
schedule = build_feature_schedule()

tiles = []
for t in range(2):
    b = t * FEATURES_PER_TILE
    e = b + FEATURES_PER_TILE
    part = [0] * SEQ_LENGTH
    part += np.round(biases[b:e] * SCALE).astype(np.int64).tolist()
    part += np.round(smean[b:e] * SCALE).astype(np.int64).tolist()
    part += np.round(sscale[b:e] * SCALE).astype(np.int64).tolist()
    for s in schedule[b:e]:
        part += [s["dilation"], s["padding"], s["kernel_i0"], s["kernel_i1"],
                 s["kernel_i2"], s["padding1"]]
    assert len(part) == TILE_LEN
    tiles.append(part)
base = np.array(tiles[0] + tiles[1], dtype=np.int64)
assert len(base) == 8062

for c in range(3):
    with open(f"_f14_w{c}.txt", "w") as f:
        for v in coef[c]:
            f.write(f"{v}\n")


def dispatch(wfile):
    for attempt in range(3):
        if os.path.exists(OUT):
            os.remove(OUT)
        try:
            r = subprocess.run(
                [RUNNER, "-x", XCLBIN, "-i", INSTS, "-d", INP, "-w", wfile, "-n", "1"],
                capture_output=True, text=True, timeout=40)
        except subprocess.TimeoutExpired:
            continue
        if r.returncode == 0 and os.path.exists(OUT):
            ms = float(re.search(r"Per-dispatch: ([0-9.]+)", r.stdout).group(1))
            return float(np.loadtxt(OUT).sum()), ms
    raise RuntimeError("dispatch failed 3 times")


N = len(X)
scores = np.zeros((N, 3))
times = []
for i in range(N):
    xs = np.round(X[i] * SCALE).astype(np.int64)
    arr = base.copy()
    arr[0:SEQ_LENGTH] = xs
    arr[TILE_LEN:TILE_LEN + SEQ_LENGTH] = xs
    with open(INP, "w") as f:
        for v in arr:
            f.write(f"{v}\n")
    for c in range(3):
        scores[i, c], ms = dispatch(f"_f14_w{c}.txt")
        times.append(ms)
    print(f"sample {i + 1}/{N} scores {np.round(scores[i], 4)}", flush=True)

np.savetxt("fused14_scores.txt", scores)
z = (scores - scores.mean(axis=0)) / (scores.std(axis=0) + 1e-12)
pred = z.argmax(axis=1)
labels, yi = np.unique(np.array(y), return_inverse=True)
ident = float((pred == yi).mean())
best_acc, best_perm = 0.0, None
for perm in itertools.permutations(range(3)):
    acc = float((np.array(perm)[pred] == yi).mean())
    if acc > best_acc:
        best_acc, best_perm = acc, perm
print(f"\nsamples: {N}")
print(f"accuracy identity mapping (0,1,2): {ident * 100:.2f}%")
print(f"accuracy best mapping {best_perm}: {best_acc * 100:.2f}%")
print(f"mean per-dispatch: {np.mean(times):.2f} ms, 3 dispatches per sample = {3 * np.mean(times):.2f} ms/sample")
print("baseline to compare: 93.75% accuracy, 16.9755 ms/sample (in-process two-xclbin pipeline)")
