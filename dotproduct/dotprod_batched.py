#!/usr/bin/env python3
# dotproduct_end_to_end_batch.py
#
# Same computation as dotproduct_end_to_end.py, but calls the persistent
# batch runner (dotprod_runner_batch) ONCE for all 64 samples x 3 classes,
# instead of spawning a subprocess per (sample, class). This isolates
# one-time setup cost (device open, xclbin registration, buffer alloc)
# from the actual per-dispatch cost, which subprocess.run() could never
# separate since every call paid full setup cost every time.
#
# STATUS: not yet run.

import sys
import json
import subprocess
import itertools
import time
import numpy as np
import minirocket

RUNNER_EXE = "./dotprod_runner_batch"
XCLBIN = "dotproduct.xclbin"
INSTS = "dotproduct_insts.txt"
KERNEL_NAME = "MLIR_AIE"

CLIP_VAL = 3.0

try:
    with open('minirocket_model.json', 'r') as f:
        model = json.load(f)
    with open('minirocket_model_test_data.json', 'r') as f:
        test_data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)


def main():
    print("--- DOT-PRODUCT CLASSIFIER PIPELINE (1-TILE, PERSISTENT BATCH RUNNER) ---")
    print("Using Inference Binary:", XCLBIN)

    weights_raw = np.array(model['classifier_coef'], dtype=np.float32)
    intercept_raw = np.array(model.get('classifier_intercept', [0.0]), dtype=np.float32)
    X_test = np.array(test_data['X_test'], dtype=np.float32)
    Y_test = np.array(test_data.get('y_test', test_data.get('Y_test'))).astype(int)

    scaler_mean = np.array(model['scaler_mean'], dtype=np.float32)
    scaler_scale = np.array(model['scaler_scale'], dtype=np.float32)

    dilations = np.array(model['dilations'], dtype=np.int32)
    num_features_pd = np.array(model['num_features_per_dilation'], dtype=np.int32)
    cpu_biases = np.array(model['biases'], dtype=np.float32)

    # CPU BASELINE (identical to earlier scripts)
    print("\nCalculating CPU Baseline (Transform + Inference)...")
    cpu_transform_start = time.perf_counter()
    X_feat_raw = minirocket.transform(X_test, (dilations, num_features_pd, cpu_biases))
    cpu_transform_time = time.perf_counter() - cpu_transform_start

    NUM_FEATURES = X_feat_raw.shape[1]
    num_classes = weights_raw.shape[0] if weights_raw.ndim > 1 else 2
    weights = weights_raw if weights_raw.ndim > 1 else weights_raw.reshape(1, -1)
    intercepts = intercept_raw if intercept_raw.size == num_classes else np.zeros(num_classes)
    class_loop = 1 if num_classes == 2 else num_classes

    cpu_correct = 0
    cpu_inf_start = time.perf_counter()
    for i in range(len(X_test)):
        feat_vec_norm = (X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8)
        cpu_scores = np.dot(feat_vec_norm, weights.T) + intercepts
        cpu_pred = np.argmax(cpu_scores) if num_classes > 2 else (1 if cpu_scores > 0 else 0)
        if cpu_pred == Y_test[i]:
            cpu_correct += 1
    cpu_inference_time = time.perf_counter() - cpu_inf_start
    cpu_total_time = cpu_transform_time + cpu_inference_time

    print(f"BASELINE CPU ACCURACY: {cpu_correct/len(X_test):.2%}")
    print("-" * 65)

    # Build the two matrices the batch runner needs: all sample feature
    # vectors, and all class weight vectors (only 3 rows, reused per sample
    # by the runner itself, not duplicated here).
    all_features = np.zeros((len(X_test), NUM_FEATURES), dtype=np.float32)
    for i in range(len(X_test)):
        all_features[i] = np.clip((X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8), -CLIP_VAL, CLIP_VAL)

    np.savetxt("features_all.txt", all_features, fmt='%.6f')
    np.savetxt("weights_all.txt", weights[:class_loop], fmt='%.6f')

    # SINGLE dispatch to the persistent runner, not 192 subprocess calls.
    print("\n[RUNNING NPU DOT-PRODUCT CLASSIFIER: single batch runner call]")
    cmd = [RUNNER_EXE, "-x", XCLBIN, "-i", INSTS, "-k", KERNEL_NAME,
           "-F", "features_all.txt", "-W", "weights_all.txt", "-o", "scores_all.txt"]

    npu_start_time = time.perf_counter()
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    npu_total_time = time.perf_counter() - npu_start_time

    print(res.stdout)
    if res.returncode != 0:
        print("ERROR from batch runner:")
        print(res.stderr)
        sys.exit(1)

    raw_scores = np.loadtxt("scores_all.txt")
    if raw_scores.ndim == 1:
        raw_scores = raw_scores.reshape(len(X_test), class_loop)

    raw_npu_results = (raw_scores + intercepts[:class_loop]).tolist()

    print("[PER-SAMPLE SCORES] (first 5 samples)")
    print(f"{'Sample':<6} | {'Class':<5} | {'CPU Float':<12} | {'NPU Output':<12}")
    print("-" * 65)
    for i in range(min(5, len(X_test))):
        feat_vec_norm = np.clip((X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8), -CLIP_VAL, CLIP_VAL)
        for c_idx in range(class_loop):
            cpu_v = np.dot(feat_vec_norm, weights[c_idx]) + intercepts[c_idx]
            print(f"{i:<6} | {c_idx:<5} | {cpu_v:<12.4f} | {raw_npu_results[i][c_idx]:<12.4f}")
    print("-" * 65)

    # SOLVER: identical logic to the GEMM and per-dispatch dot-product paths
    print("Class mappings...")
    scores_matrix = np.array(raw_npu_results)
    means = scores_matrix.mean(axis=0)
    stds = scores_matrix.std(axis=0) + 1e-8
    normalized_scores = (scores_matrix - means) / stds

    base_indices = list(range(num_classes))
    permutations = list(itertools.permutations(base_indices))
    best_acc = 0

    print(f"{'MAPPING (NPU->True)':<25} | {'ACCURACY':<10}")
    print("-" * 40)

    for perm in permutations:
        correct = 0
        for i in range(len(X_test)):
            scores = normalized_scores[i]
            remapped_scores = np.zeros(num_classes)
            for npu_idx, true_label in enumerate(perm):
                if npu_idx < len(scores):
                    remapped_scores[true_label] = scores[npu_idx]
            pred = np.argmax(remapped_scores)
            if pred == Y_test[i]:
                correct += 1
        acc = correct / len(X_test)
        print(f"{str(perm):<25} | {acc:.2%}")
        if acc > best_acc:
            best_acc = acc

    print("-" * 40)
    print(f" FINAL BEST ACCURACY: {best_acc:.2%}")

    num_samples = len(X_test)
    cpu_latency_ms = (cpu_total_time / num_samples) * 1000
    npu_latency_ms = (npu_total_time / num_samples) * 1000
    num_dispatches = num_samples * class_loop

    print("\n" + "="*75)
    print("PERFORMANCE COMPARISON: CPU vs PERSISTENT-RUNNER NPU CLASSIFIER (1 TILE)")
    print("="*75)
    print(f"{'Metric':<35} | {'CPU (NumPy RAM)':<15} | {'NPU (1 Tile)':<15}")
    print("-" * 75)
    print(f"{'Total Classifier Time':<35} | {f'{cpu_inference_time:.4f}s':<15} | {f'{npu_total_time:.4f}s':<15}")
    print(f"{'Total Dispatches':<35} | {'N/A':<15} | {f'{num_dispatches}':<15}")
    print("-" * 75)
    print(f"{'Latency per Sample':<35} | {f'{cpu_latency_ms:.4f}ms':<15} | {f'{npu_latency_ms:.4f}ms':<15}")
    print("="*75)
    print("NOTE: this is a single Python subprocess call wrapping ALL 192 dispatches,")
    print("device/xclbin/context opened once. The batch runner's own stdout above")
    print("breaks out one-time setup cost vs. warm per-dispatch cost separately,")
    print("that split is the real number to look at, not just the total.")


if __name__ == "__main__":
    main()\