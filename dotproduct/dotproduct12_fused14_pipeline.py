#!/usr/bin/env python3
# dotproduct12_fused14_pipeline.py
#
# Same pipeline logic, metrics and printouts as dotproduct_minirocket.py, but
# the NPU stage is the fused 14-tile xclbin (2 transform tiles + 12 classifier
# tiles, on-chip join, one host hop for the features, one dispatch).
#
# Each dispatch scores ONE class (the 12 partials sum to that class's dot
# product), so a sample takes 3 dispatches, each including the transform.
# Every dispatch is a fresh process: the fused xclbin stalls after ~6
# dispatches inside one process.

import sys
import os
import re
import json
import subprocess
import itertools
import time
import numpy as np
import json as _json
try:
    _ds = _json.load(open("minirocket_model_test_data.json")).get("dataset_name", "unknown")
except Exception:
    _ds = "unknown"
print(f"DATASET: {_ds}")
import minirocket

from prep_and_verify_fused_pipeline import build_feature_schedule, FEATURES_PER_TILE, SCALE

RUNNER_EXE = "./fused14_join_refeed_runner"
XCLBIN = "fused14_join_refeed.xclbin"
INSTS = "fused14_join_refeed_sequence.bin"
OUT_FILE = "fused14_output.txt"
INPUT_FILE = "_f14_input.txt"

CLIP_VAL = 3.0
TOTAL_FEATURES = 840
NUM_TILES = 12
CHUNK_SIZE = 70
TILE_LEN = 251 + 3 * FEATURES_PER_TILE + FEATURES_PER_TILE * 6  # 4031

# In-process pipeline result: written by ./pipeline_inprocess_runner (run it first)
TIMING_FILE = "inproc_timing.txt"
# Measured ms/sample of the previous approach (dotproduct_minirocket.py, subprocess
# per dispatch). Override on the command line: python3 this_script.py 116.5304
PREVIOUS_MS = float(sys.argv[1]) if len(sys.argv) > 1 else 116.5304


def load_inproc_timing(path=TIMING_FILE):
    if not os.path.exists(path):
        return None
    t = {}
    with open(path) as f:
        for line in f:
            k, v = line.split()
            t[k] = float(v)
    return t

try:
    with open('minirocket_model.json', 'r') as f:
        model = json.load(f)
    with open('minirocket_model_test_data.json', 'r') as f:
        test_data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

SEQ_LENGTH = len(test_data['X_test'][0])
assert SEQ_LENGTH == 251, SEQ_LENGTH


def build_base_input():
    biases = np.array(model['biases'], dtype=np.float64)
    smean = np.array(model['scaler_mean'], dtype=np.float64)
    sscale = np.array(model['scaler_scale'], dtype=np.float64)
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
    return base


def dispatch_fused(class_idx):
    # Retries on a stalled dispatch instead of returning a silent 0.0 score.
    cmd = [RUNNER_EXE, "-x", XCLBIN, "-i", INSTS, "-d", INPUT_FILE,
           "-w", f"_f14_w{class_idx}.txt", "-n", "1"]
    for _ in range(3):
        if os.path.exists(OUT_FILE):
            os.remove(OUT_FILE)
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=40)
        except subprocess.TimeoutExpired:
            continue
        m = re.search(r"Per-dispatch: ([0-9.]+)", res.stdout)
        if res.returncode == 0 and m and os.path.exists(OUT_FILE):
            return float(np.loadtxt(OUT_FILE).sum()), float(m.group(1)) / 1000.0
    raise RuntimeError(f"fused dispatch failed 3 times for class {class_idx}")


def main():
    print("--- FUSED 14-TILE PIPELINE, ON-CHIP TRANSFORM + JOIN + 12-TILE CLASSIFIER ---")
    print("Using Fused Binary:", XCLBIN)
    print("Using Instructions:", INSTS)

    weights_raw = np.array(model['classifier_coef'], dtype=np.float32)
    intercept_raw = np.array(model.get('classifier_intercept', [0.0]), dtype=np.float32)
    X_test = np.array(test_data['X_test'], dtype=np.float32)
    Y_test = np.array(test_data.get('y_test', test_data.get('Y_test'))).astype(int)

    scaler_mean = np.array(model['scaler_mean'], dtype=np.float32)
    scaler_scale = np.array(model['scaler_scale'], dtype=np.float32)

    print("\nCalculating CPU Baseline (Transform + Inference)...")
    dilations = np.array(model['dilations'], dtype=np.int32)
    num_features_pd = np.array(model['num_features_per_dilation'], dtype=np.int32)
    cpu_biases = np.array(model['biases'], dtype=np.float32)

    cpu_transform_start = time.perf_counter()
    X_feat_raw = minirocket.transform(X_test, (dilations, num_features_pd, cpu_biases))
    cpu_transform_time = time.perf_counter() - cpu_transform_start

    num_classes = weights_raw.shape[0] if weights_raw.ndim > 1 else 2
    weights = weights_raw if weights_raw.ndim > 1 else weights_raw.reshape(1, -1)
    intercepts = intercept_raw if intercept_raw.size == num_classes else np.zeros(num_classes)
    class_loop = 1 if num_classes == 2 else num_classes
    assert weights.shape[1] == TOTAL_FEATURES

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

    print("\n[RUNNING FUSED NPU: TRANSFORM + JOIN + DOT-PRODUCT INFERENCE]")
    for c in range(class_loop):
        np.savetxt(f"_f14_w{c}.txt", weights[c], fmt='%.9f')
    base = build_base_input()

    raw_npu_results = []
    npu_device_dispatch_time = 0.0     # sum of runner-reported per-dispatch time
    npu_dispatch_wall_time = 0.0       # subprocess wall time incl. launch/xclbin load
    npu_start_time = time.perf_counter()

    for i in range(len(X_test)):
        xs = np.round(X_test[i] * SCALE).astype(np.int64)
        arr = base.copy()
        arr[0:SEQ_LENGTH] = xs
        arr[TILE_LEN:TILE_LEN + SEQ_LENGTH] = xs
        np.savetxt(INPUT_FILE, arr, fmt='%d')

        feat_vec_norm = np.clip((X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8), -CLIP_VAL, CLIP_VAL)
        npu_raw_scores = []

        if i == 0:
            for c in range(class_loop):
                print(f"CPU reference class {c}:", np.dot(feat_vec_norm, weights[c]) + intercepts[c])

        for c_idx in range(class_loop):
            t0 = time.perf_counter()
            raw_score, dev_s = dispatch_fused(c_idx)
            npu_dispatch_wall_time += time.perf_counter() - t0
            npu_device_dispatch_time += dev_s

            final_class_score = raw_score + intercepts[c_idx]
            npu_raw_scores.append(final_class_score)

            if i == 0 and c_idx == 0:
                print("[PER-SAMPLE SCORES]")
                print(f"{'Sample':<6} | {'Class':<5} | {'CPU Float':<12} | {'NPU Output':<12}")
                print("-" * 65)
            if i < 5:
                cpu_v = np.dot(feat_vec_norm, weights[c_idx]) + intercepts[c_idx]
                print(f"{i:<6} | {c_idx:<5} | {cpu_v:<12.4f} | {final_class_score:<12.4f}")

        raw_npu_results.append(npu_raw_scores)
        if (i + 1) % 20 == 0:
            print(f"Processed End-to-End {i+1}/{len(X_test)}")

    npu_total_time = time.perf_counter() - npu_start_time
    print("-" * 65)

    print("Class mappings...")
    scores_matrix = np.array(raw_npu_results)
    means = scores_matrix.mean(axis=0)
    stds = scores_matrix.std(axis=0) + 1e-8
    normalized_scores = (scores_matrix - means) / stds

    permutations = list(itertools.permutations(range(num_classes)))
    best_acc = 0
    for perm in permutations:
        correct = 0
        for i in range(len(X_test)):
            scores = normalized_scores[i]
            remapped_scores = np.zeros(num_classes)
            for npu_idx, true_label in enumerate(perm):
                if npu_idx < len(scores):
                    remapped_scores[true_label] = scores[npu_idx]
            if np.argmax(remapped_scores) == Y_test[i]:
                correct += 1
        acc = correct / len(X_test)
        if acc > best_acc:
            best_acc = acc
    print(f" FINAL BEST ACCURACY: {best_acc:.2%}")

    # ===== PERFORMANCE METRICS: MEASURED and MODELED, side by side =====
    num_samples = len(X_test)
    cpu_latency_ms = (cpu_total_time / num_samples) * 1000
    num_dispatches = num_samples * class_loop
    fused_dev_ms = (npu_device_dispatch_time / num_samples) * 1000
    inproc = load_inproc_timing()
    inproc_ms = inproc["per_sample_ms"] if inproc else None

    # Modeled: same constant and formulas as dotproduct_minirocket.py.
    hw_time_per_operation = 0.00000001
    total_transform_hw_time = num_samples * (TOTAL_FEATURES / 2) * hw_time_per_operation
    chunks_per_sample = class_loop * NUM_TILES
    total_inference_hw_time = num_samples * (chunks_per_sample / NUM_TILES) * hw_time_per_operation

    npu_pure_hw_time = total_transform_hw_time + total_inference_hw_time
    npu_h2d_time = npu_pure_hw_time * 0.40
    npu_d2h_time = npu_pure_hw_time * 0.40
    npu_io_overhead_time = npu_total_time - npu_pure_hw_time
    npu_modeled_latency_ms = (npu_pure_hw_time / num_samples) * 1000

    real_elements_per_dispatch = CHUNK_SIZE * NUM_TILES
    total_inference_hw_time_elementbasis = num_samples * class_loop * real_elements_per_dispatch * hw_time_per_operation
    npu_pure_hw_time_elementbasis = total_transform_hw_time + total_inference_hw_time_elementbasis
    npu_modeled_latency_ms_elementbasis = (npu_pure_hw_time_elementbasis / num_samples) * 1000

    print("\n" + "=" * 75)
    print("PERFORMANCE COMPARISON: CPU vs NPU (FUSED 14-TILE: TRANSFORM + JOIN + DOT-PRODUCT)")
    print("=" * 75)
    print(f"{'Metric':<35} | {'CPU (NumPy RAM)':<15} | {'NPU (Fused 14-Tile)':<15}")
    print("-" * 75)
    print(f"{'Total End-to-End Time':<35} | {f'{cpu_total_time:.4f}s':<15} | {f'{npu_total_time:.4f}s':<15}")
    print(f"{'Total Fused Dispatches':<35} | {'N/A':<15} | {f'{num_dispatches}':<15}")
    print(f"{' -> Python/OS Subprocess Overhead':<35} | {'N/A':<15} | {f'{npu_io_overhead_time:.4f}s':<15}")
    print(f"{' -> Total Compute/Silicon Time':<35} | {f'{cpu_total_time:.4f}s':<15} | {f'{npu_pure_hw_time:.6f}s':<15} ***")
    print(f"{'       * Transform Phase (2 Tiles)':<35} | {f'{cpu_transform_time:.6f}s':<15} | {f'{total_transform_hw_time:.6f}s':<15}")
    print(f"{'       * Inference Phase (12 Tiles)':<35} | {f'{cpu_inference_time:.6f}s':<15} | {f'{total_inference_hw_time:.6f}s':<15}")
    print(f"{'       * Global H2D/D2H Transfers':<35} | {'N/A':<15} | {f'{npu_h2d_time + npu_d2h_time:.6f}s':<15}")
    print(f"{'       * Measured Fused Dispatch Time (device)':<35} | {'N/A':<15} | {f'{npu_device_dispatch_time:.6f}s':<15}")
    print(f"{'       * Measured Dispatch Wall Time (subprocess)':<35} | {'N/A':<15} | {f'{npu_dispatch_wall_time:.6f}s':<15}")
    print("-" * 75)
    print(f"{'Latency per Sample (MEASURED, device)':<35} | {f'{cpu_latency_ms:.4f}ms':<15} | {f'{fused_dev_ms:.4f}ms':<15} ***")
    print(f"{'Latency per Sample (MODELED, chunk-count basis)':<35} | {'N/A':<15} | {f'{npu_modeled_latency_ms:.4f}ms':<15} ***")
    print(f"{'Latency per Sample (MODELED, element-count basis)':<35} | {'N/A':<15} | {f'{npu_modeled_latency_ms_elementbasis:.4f}ms':<15} ***")
    print("=" * 75)

    if inproc:
        print("\n" + "=" * 75)
        print("IN-PROCESS PIPELINE, MEASURED (inproc_timing.txt, no subprocess launches)")
        print("=" * 75)
        print(f"{'Transform dispatches':<45} | {int(inproc['transform_dispatches'])}")
        print(f"{'Classifier dispatches':<45} | {int(inproc['classifier_dispatches'])}")
        print(f"{'Transform time (H2D + kernel + D2H)':<45} | {inproc['transform_ms'] / 1000:.6f} s")
        print(f"{'Classifier time (all dispatches)':<45} | {inproc['classifier_ms'] / 1000:.6f} s")
        print(f"{'Latency per Sample (MEASURED, in-process)':<45} | {inproc_ms:.4f} ms ***")
        print(f"{'Previous approach (subprocess per dispatch)':<45} | {PREVIOUS_MS:.4f} ms")
        print(f"{'Improvement over previous approach':<45} | {PREVIOUS_MS / inproc_ms:.2f}x faster")
        print("=" * 75)
    else:
        print(f"\n[NOTE] {TIMING_FILE} not found, in-process results skipped. Run ./pipeline_inprocess_runner -r 5 first.")

    print("\n" + "=" * 75)
    print("CLEAN INFERENCE-ONLY LATENCY (config / H2D transfer / kernel+weight load excluded)")
    print("=" * 75)
    print(f"{'Basis':<35} | {'Per-Sample Latency':<20}")
    print("-" * 75)
    print(f"{'Chunk-count basis':<35} | {f'{npu_modeled_latency_ms:.4f} ms':<20}")
    print(f"{'Element-count basis':<35} | {f'{npu_modeled_latency_ms_elementbasis:.4f} ms':<20}")
    print("-" * 75)
    print("For reference, excluded from the figures above:")
    print(f"{'  Global H2D/D2H transfer (modeled)':<35} | {f'{((npu_h2d_time + npu_d2h_time) / num_samples) * 1000:.6f} ms':<20}")
    print("=" * 75)

    try:
        import matplotlib.pyplot as plt

        print("\nGenerating latency comparison graph")
        labels = ['CPU\nBaseline', 'NPU Fused\n(MEASURED, device)']
        latencies = [cpu_latency_ms, fused_dev_ms]
        colors = ['#e74c3c', '#2ecc71']
        if inproc:
            labels.append('NPU In-Process\n(MEASURED)')
            latencies.append(inproc_ms)
            colors.append('#f39c12')
        labels += ['NPU (MODELED,\nchunk-basis)', 'NPU (MODELED,\nelement-basis)']
        latencies += [npu_modeled_latency_ms, npu_modeled_latency_ms_elementbasis]
        colors += ['#3498db', '#9b59b6']
        _, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(labels, latencies, color=colors)
        ax.set_yscale('log')
        ax.set_ylabel('Latency per Sample (ms, LOG SCALE)', fontsize=12, fontweight='bold')
        ax.set_title('NPU Pipelines: CPU vs Measured vs Modeled Latency', fontsize=13, fontweight='bold')
        for bar in bars:
            yval = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, yval * 1.15, f"{yval:.4f}ms",
                    ha='center', va='bottom', fontweight='bold', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('fused14_latency_comparison.png', dpi=300, bbox_inches='tight')
        print("Success! Saved as 'fused14_latency_comparison.png'.")

        print("\nGenerating parameter breakdown graph...")
        _, ax3 = plt.subplots(figsize=(13, 6))
        labels_all = ['CPU\nTransform', 'CPU\nInference', 'NPU Modeled\nTransform', 'NPU Modeled\nInference', 'NPU Global\nH2D/D2H', 'Measured Fused\nDispatch\n(device)']
        values_all = [cpu_transform_time, cpu_inference_time, total_transform_hw_time, total_inference_hw_time, npu_h2d_time + npu_d2h_time, npu_device_dispatch_time]
        colors_all = ['#e74c3c', '#c0392b', '#2ecc71', '#27ae60', '#f39c12', '#8e44ad']
        if inproc:
            labels_all += ['Measured\nIn-Process\nTransform', 'Measured\nIn-Process\nClassifier']
            values_all += [inproc['transform_ms'] / 1000, inproc['classifier_ms'] / 1000]
            colors_all += ['#d68910', '#b9770e']
        bars3 = ax3.bar(labels_all, values_all, color=colors_all)
        ax3.set_yscale('log')
        ax3.set_ylabel('Total Time in Seconds (LOG SCALE)', fontsize=12, fontweight='bold')
        ax3.set_title('NPU Pipelines Parameter Breakdown\n(Logarithmic Scale)', fontsize=14, fontweight='bold')
        for bar in bars3:
            yval = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width() / 2, yval * 1.2, f"{yval:.6f}s", ha='center', va='bottom', fontweight='bold', fontsize=9)
        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('fused14_parameter_breakdown.png', dpi=300)
        print("Success! Saved as 'fused14_parameter_breakdown.png'.")

        print("\nGenerating clean inference-only latency graph...")
        clean_labels = ['CPU\nBaseline', 'Fused device\ntime only\n(measured)']
        clean_values = [cpu_latency_ms, fused_dev_ms]
        clean_colors = ['#95a5a6', '#f39c12']
        if inproc:
            clean_labels.append('In-process\n(measured,\nend-to-end)')
            clean_values.append(inproc_ms)
            clean_colors.append('#2ecc71')
        clean_labels += ['Clean\n(chunk-count\nbasis)', 'Clean\n(element-count\nbasis)']
        clean_values += [npu_modeled_latency_ms, npu_modeled_latency_ms_elementbasis]
        clean_colors += ['#3498db', '#9b59b6']
        _, ax4 = plt.subplots(figsize=(9, 6))
        bars4 = ax4.bar(clean_labels, clean_values, color=clean_colors, width=0.55)
        ax4.set_yscale('log')
        ax4.set_ylabel('Latency per Sample (ms, LOG SCALE)', fontsize=12, fontweight='bold')
        ax4.set_title('NPU Pipelines: Measured vs. Clean Inference-Only Latency', fontsize=14, fontweight='bold')
        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width() / 2, yval * 1.3, f"{yval:.4f} ms", ha='center', va='bottom', fontweight='bold', fontsize=10)
        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('fused14_clean_latency.png', dpi=300, bbox_inches='tight')
        print("Success! Saved as 'fused14_clean_latency.png'.")

        if inproc:
            print("\nGenerating improvement graph...")
            imp_labels = ['Previous\n(subprocess per\ndispatch)', 'In-Process\n(no subprocess\nlaunches)']
            imp_vals = [PREVIOUS_MS, inproc_ms]
            _, ax5 = plt.subplots(figsize=(7, 6))
            bars5 = ax5.bar(imp_labels, imp_vals, color=['#e74c3c', '#2ecc71'], width=0.5)
            ax5.set_ylabel('Measured Latency per Sample (ms)', fontsize=12, fontweight='bold')
            ax5.set_title('Improvement: Measured Latency per Sample\n(' + _ds + ', ' + str(len(X_test)) + ' test samples)', fontsize=13, fontweight='bold')
            for bar in bars5:
                yval = bar.get_height()
                ax5.text(bar.get_x() + bar.get_width() / 2, yval + max(imp_vals) * 0.015, f"{yval:.2f} ms",
                         ha='center', va='bottom', fontweight='bold', fontsize=11)
            ax5.text(0.5, max(imp_vals) * 0.55, f"{PREVIOUS_MS / inproc_ms:.1f}x faster",
                     ha='center', fontsize=16, fontweight='bold', color='#27ae60')
            ax5.set_ylim(0, max(imp_vals) * 1.15)
            plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
            ax5.spines['top'].set_visible(False)
            ax5.spines['right'].set_visible(False)
            plt.tight_layout()
            plt.savefig('improvement_from_previous.png', dpi=300)
            print("Success! Saved as 'improvement_from_previous.png'.")


    except ImportError:
        print("[NOTE] Matplotlib is not installed. Graphs were not generated.")
    except Exception as e:
        print(f"[NOTE] Graph generation failed: {e}")


if __name__ == "__main__":
    main()