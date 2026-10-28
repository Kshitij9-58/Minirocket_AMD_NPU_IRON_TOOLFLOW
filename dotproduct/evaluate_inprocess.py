#!/usr/bin/env python3
"""
evaluate_inprocess.py

Reads inproc_scores.txt (raw per-class dot products from the NPU, one line per
sample) and inproc_timing.txt written by pipeline_inprocess_runner, then
reproduces the validated driver's scoring:

  NPU class score = raw dot product + intercept
  solver          = z-score each class column across samples, try every
                    class permutation, report the best accuracy
                    (same logic as dotproduct_minirocket.py)

Run from the dotproduct directory, after the runner.
"""

import itertools
import json

import numpy as np

import minirocket

CLIP_VAL = 3.0


def load_timing(path="inproc_timing.txt"):
    t = {}
    with open(path) as f:
        for line in f:
            k, v = line.split()
            t[k] = float(v)
    return t


def main():
    with open("minirocket_model.json") as f:
        model = json.load(f)
    with open("minirocket_model_test_data.json") as f:
        test = json.load(f)

    X_test = np.array(test["X_test"], dtype=np.float32)
    Y_test = np.array(test["y_test"])
    weights = np.array(model["classifier_coef"], dtype=np.float32)
    intercepts = np.array(model["classifier_intercept"], dtype=np.float32)
    scaler_mean = np.array(model["scaler_mean"], dtype=np.float32)
    scaler_scale = np.array(model["scaler_scale"], dtype=np.float32)
    dilations = np.array(model["dilations"], dtype=np.int32)
    nfpd = np.array(model["num_features_per_dilation"], dtype=np.int32)
    biases = np.array(model["biases"], dtype=np.float32)
    num_classes = weights.shape[0]
    num_samples = len(X_test)

    print("--- DOT-PRODUCT PIPELINE, FULL NPU TRANSFORM + 12-TILE CLASSIFIER (IN-PROCESS DISPATCH) ---")

    # CPU baseline, same as the validated driver (unclipped features)
    X_feat_raw = minirocket.transform(X_test, (dilations, nfpd, biases))
    cpu_feat = (X_feat_raw - scaler_mean) / (scaler_scale + 1e-8)
    cpu_scores_all = cpu_feat @ weights.T + intercepts
    cpu_pred = np.argmax(cpu_scores_all, axis=1)
    print(f"BASELINE CPU ACCURACY: {np.mean(cpu_pred == Y_test):.2%}")
    print("-" * 65)

    raw = np.loadtxt("inproc_scores.txt", ndmin=2)
    if raw.shape != (num_samples, num_classes):
        raise SystemExit(f"inproc_scores.txt has shape {raw.shape}, expected {(num_samples, num_classes)}")
    npu_scores = raw + intercepts

    print("[PER-SAMPLE SCORES]")
    print(f"{'Sample':<6} | {'Class':<5} | {'CPU Float':<12} | {'NPU Output':<12}")
    print("-" * 65)
    clipped = np.clip(cpu_feat, -CLIP_VAL, CLIP_VAL)
    for i in range(min(5, num_samples)):
        for c in range(num_classes):
            cpu_v = float(np.dot(clipped[i], weights[c]) + intercepts[c])
            print(f"{i:<6} | {c:<5} | {cpu_v:<12.4f} | {npu_scores[i, c]:<12.4f}")
    print("-" * 65)

    # Solver, identical to dotproduct_minirocket.py
    print("Class mappings...")
    scores_matrix = np.array(npu_scores)
    means = scores_matrix.mean(axis=0)
    stds = scores_matrix.std(axis=0) + 1e-8
    normalized_scores = (scores_matrix - means) / stds

    permutations = list(itertools.permutations(range(num_classes)))
    best_acc = 0.0
    print(f"{'MAPPING (NPU->True)':<25} | {'ACCURACY':<10}")
    print("-" * 40)
    for perm in permutations:
        correct = 0
        for i in range(num_samples):
            scores = normalized_scores[i]
            remapped = np.zeros(num_classes)
            for npu_idx, true_label in enumerate(perm):
                if npu_idx < len(scores):
                    remapped[true_label] = scores[npu_idx]
            if np.argmax(remapped) == Y_test[i]:
                correct += 1
        acc = correct / num_samples
        print(f"{str(perm):<25} | {acc:.2%}")
        best_acc = max(best_acc, acc)
    print("-" * 40)
    print(f" FINAL BEST ACCURACY: {best_acc:.2%}")

    # Measured timing from the runner
    t = load_timing()
    n = int(t["n_samples"])
    print()
    print("=" * 75)
    print("MEASURED (in-process dispatch, no subprocess launches)")
    print("=" * 75)
    print(f"{'Samples':<40} | {n}")
    print(f"{'Transform dispatches':<40} | {int(t['transform_dispatches'])}")
    print(f"{'Classifier dispatches':<40} | {int(t['classifier_dispatches'])}")
    print(f"{'Transform time (H2D + kernel + D2H)':<40} | {t['transform_ms'] / 1000:.6f} s")
    print(f"{'Classifier time (all dispatches)':<40} | {t['classifier_ms'] / 1000:.6f} s")
    print(f"{'Total end-to-end loop time':<40} | {t['loop_ms'] / 1000:.6f} s")
    print(f"{'Latency per sample (MEASURED)':<40} | {t['per_sample_ms']:.4f} ms")
    print(f"{'One-time setup (device + 2 xclbins)':<40} | {t['setup_ms'] / 1000:.6f} s  (excluded from latency)")
    print("=" * 75)


if __name__ == "__main__":
    main()
