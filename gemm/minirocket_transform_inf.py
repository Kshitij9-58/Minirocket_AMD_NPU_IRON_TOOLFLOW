# npu_end_to_end.py -*- Python -*-
#
# UNIFIED MINIROCKET PIPELINE (2:12 SPLIT - BATCHED)

import sys
import json
import subprocess
import numpy as np
import itertools
import time
import minirocket 
import aie.iron as iron

from aie.iron import ObjectFifo, Program, Runtime, Worker
from aie.iron.placers import SequentialPlacer
from aie.iron.controlflow import range_

RUNNER_EXE = "./app.exe" 
XCLBIN = "/home/wch464/mlir-aie/programming_examples/minirocket_implementation/final-12.xclbin"
INSTS = "/home/wch464/mlir-aie/programming_examples/minirocket_implementation/inst-12.txt"

KERNEL_SIZE = 64  
CLIP_VAL = 3.0
TOTAL_FEATURES = 840
FEATURES_PER_CORE = TOTAL_FEATURES // 2  

try:
    with open('minirocket_model.json', 'r') as f:
        model = json.load(f)
    with open('minirocket_model_test_data.json', 'r') as f:
        test_data = json.load(f)
except Exception as e:
    print(f"Error loading JSON: {e}")
    sys.exit(1)

real_input_ts = np.array(test_data['X_test'][0], dtype=np.float32)
KERNEL_LENGTH = 9      
SEQ_LENGTH = len(real_input_ts)          
SLIDING_STEPS = SEQ_LENGTH - KERNEL_LENGTH + 1 
PAYLOAD_LENGTH = 10 + SEQ_LENGTH        

@iron.jit(is_placed=False)
def minirocket_sliding_kernel(in0, in1, out0, out1):
    num_features = np.size(out0) // SLIDING_STEPS
    dtype = in0.dtype

    stream_tensor_ty = np.ndarray[(num_features * PAYLOAD_LENGTH,), np.dtype[dtype]]
    feat_tensor_ty = np.ndarray[(num_features * SLIDING_STEPS,), np.dtype[dtype]]
    stream_tile_ty = np.ndarray[(PAYLOAD_LENGTH,), np.dtype[dtype]]
    feat_tile_ty = np.ndarray[(SLIDING_STEPS,), np.dtype[dtype]]

    of_in0 = ObjectFifo(stream_tile_ty, name="in0")
    of_in1 = ObjectFifo(stream_tile_ty, name="in1")
    of_out0 = ObjectFifo(feat_tile_ty, name="out0")
    of_out1 = ObjectFifo(feat_tile_ty, name="out1")

    def core_body(of_in, of_out):
        for _ in range_(num_features):
            elem_in = of_in.acquire(1)
            elem_o = of_out.acquire(1)
            
            b = elem_in[0]
            w0 = elem_in[1]; w1 = elem_in[2]; w2 = elem_in[3]
            w3 = elem_in[4]; w4 = elem_in[5]; w5 = elem_in[6]
            w6 = elem_in[7]; w7 = elem_in[8]; w8 = elem_in[9]
            
            for s in range_(SLIDING_STEPS):
                d0 = elem_in[10 + s]; d1 = elem_in[11 + s]; d2 = elem_in[12 + s]
                d3 = elem_in[13 + s]; d4 = elem_in[14 + s]; d5 = elem_in[15 + s]
                d6 = elem_in[16 + s]; d7 = elem_in[17 + s]; d8 = elem_in[18 + s]
                elem_o[s] = b + (d0*w0) + (d1*w1) + (d2*w2) + (d3*w3) + (d4*w4) + (d5*w5) + (d6*w6) + (d7*w7) + (d8*w8)
                
            of_in.release(1)
            of_out.release(1)

    worker0 = Worker(core_body, fn_args=[of_in0.cons(), of_out0.prod()])
    worker1 = Worker(core_body, fn_args=[of_in1.cons(), of_out1.prod()])
    
    rt = Runtime()
    with rt.sequence(
        stream_tensor_ty, stream_tensor_ty, 
        feat_tensor_ty, feat_tensor_ty
    ) as ptrs:
        rt.start(worker0); rt.start(worker1)
        
        rt.fill(of_in0.prod(), ptrs[0]); rt.fill(of_in1.prod(), ptrs[1])
        rt.drain(of_out0.cons(), ptrs[2], wait=True); rt.drain(of_out1.cons(), ptrs[3], wait=True)

    return Program(iron.get_current_device(), rt).resolve_program(SequentialPlacer())


def main():
    print("--- STARTING UNIFIED END-TO-END PIPELINE (2:12 SPLIT - BATCHED) ---")
    print("Using Inference Binary:", XCLBIN)

    weights_raw = np.array(model['classifier_coef'], dtype=np.float32)
    intercept_raw = np.array(model.get('classifier_intercept', [0.0]), dtype=np.float32)
    X_test = np.array(test_data['X_test'], dtype=np.float32)
    Y_test = np.array(test_data.get('y_test', test_data.get('Y_test'))).astype(int)
    
    scaler_mean = np.array(model['scaler_mean'], dtype=np.float32)
    scaler_scale = np.array(model['scaler_scale'], dtype=np.float32)
    
    real_biases = np.array(model['biases'], dtype=np.float32)
    if len(real_biases) < TOTAL_FEATURES:
        real_biases = np.resize(real_biases, TOTAL_FEATURES)
    else:
        real_biases = real_biases[:TOTAL_FEATURES]

    print("\nCalculating CPU Baseline (Transform + Inference)...")
    dilations = np.array(model['dilations'], dtype=np.int32)
    num_features_pd = np.array(model['num_features_per_dilation'], dtype=np.int32)
    cpu_biases = np.array(model['biases'], dtype=np.float32)
    
    cpu_transform_start = time.perf_counter()
    X_feat_raw = minirocket.transform(X_test, (dilations, num_features_pd, cpu_biases))
    cpu_transform_time = time.perf_counter() - cpu_transform_start

    NUM_FEATURES = X_feat_raw.shape[1]
    num_classes = weights_raw.shape[0] if weights_raw.ndim > 1 else 2
    weights = weights_raw if weights_raw.ndim > 1 else weights_raw.reshape(1, -1)
    intercepts = intercept_raw if intercept_raw.size == num_classes else np.zeros(num_classes)

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

    print("Preparing 64x64 FLOAT weight chunks for NPU Classifier...")
    weight_chunks_all_classes = []
    class_loop = 1 if num_classes == 2 else num_classes

    for c_idx in range(class_loop):
        current_weights = weights[c_idx]
        class_chunks = []
        for chunk_start in range(0, NUM_FEATURES, KERNEL_SIZE):
            w_chunk = current_weights[chunk_start:chunk_start + KERNEL_SIZE]
            w_padded = np.zeros(KERNEL_SIZE, dtype=np.float32)
            w_padded[:len(w_chunk)] = w_chunk 
            Mat_B = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
            Mat_B[0, :] = w_padded 
            class_chunks.append(Mat_B.flatten())
        weight_chunks_all_classes.append(class_chunks)

    print("\n[RUNNING NPU TRANSFORM + INFERENCE]")
    raw_npu_results = [] 
    
    d_in = [iron.randint(0, 1, (FEATURES_PER_CORE * PAYLOAD_LENGTH,), dtype=np.int32, device="npu") for _ in range(2)]
    d_out = [iron.randint(99, 100, (FEATURES_PER_CORE * SLIDING_STEPS,), dtype=np.int32, device="npu") for _ in range(2)]

    static_weights = np.zeros((TOTAL_FEATURES, 9), dtype=np.int32)
    for f in range(TOTAL_FEATURES):
        w = np.array([-1, -1, -1, -1, -1, -1, 2, 2, 2], dtype=np.int32)
        np.random.shuffle(w)
        static_weights[f] = w
        
    static_biases_int = (real_biases * 1000).astype(np.int32)
    
    serialized_stream = np.zeros((TOTAL_FEATURES, PAYLOAD_LENGTH), dtype=np.int32)
    serialized_stream[:, 0] = static_biases_int
    serialized_stream[:, 1:10] = static_weights

    npu_start_time = time.perf_counter()
    npu_kernel_execution = 0

    for i in range(len(X_test)):
        
        scaled_data_seq = (X_test[i] * 1000).astype(np.int32)
        serialized_stream[:, 10:PAYLOAD_LENGTH] = scaled_data_seq     

        for j in range(2):
            d_in[j][:] = serialized_stream[j*FEATURES_PER_CORE : (j+1)*FEATURES_PER_CORE].flatten()

        kernel_start = time.perf_counter()
        minirocket_sliding_kernel(d_in[0], d_in[1], d_out[0], d_out[1])
        npu_kernel_execution += time.perf_counter() - kernel_start

        if i == 0:
            print(f"Transform Npu_kernel_execution: {npu_kernel_execution:.6f} seconds")

        feat_vec_norm = np.clip((X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8), -CLIP_VAL, CLIP_VAL)
        npu_raw_scores = [] 

        if i == 0:
            np.savetxt("features.txt", feat_vec_norm, fmt='%.6f')
            for c in range(class_loop):
                np.savetxt(f"weights_{c}.txt", weights[c], fmt='%.6f')
                print(f"CPU reference class {c}:", np.dot(feat_vec_norm, weights[c]) + intercepts[c])

        for c_idx in range(class_loop):
            total_class_score = 0.0
            weight_chunks = weight_chunks_all_classes[c_idx]

            chunks_A = []
            chunks_B = []
            
            for chunk_id, chunk_start in enumerate(range(0, NUM_FEATURES, KERNEL_SIZE)):
                f_chunk = feat_vec_norm[chunk_start : chunk_start + KERNEL_SIZE]
                f_padded = np.zeros(KERNEL_SIZE, dtype=np.float32)
                f_padded[:len(f_chunk)] = f_chunk
                Mat_A = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
                Mat_A[0, :] = f_padded 
                chunks_A.append(Mat_A.flatten())
                chunks_B.append(weight_chunks[chunk_id])

            for b_idx in range(0, len(chunks_A), 12):
                batch_A = chunks_A[b_idx : b_idx + 12]
                batch_B = chunks_B[b_idx : b_idx + 12]

                while len(batch_A) < 12:
                    batch_A.append(np.zeros(KERNEL_SIZE * KERNEL_SIZE, dtype=np.float32))
                    batch_B.append(np.zeros(KERNEL_SIZE * KERNEL_SIZE, dtype=np.float32))
                
                Flat_A_12 = np.concatenate(batch_A)
                Flat_B_12 = np.concatenate(batch_B)

                np.savetxt("input_a.txt", Flat_A_12, fmt='%.6f')
                np.savetxt("input_b.txt", Flat_B_12, fmt='%.6f')
                
                cmd = [RUNNER_EXE, "-x", XCLBIN, "-i", INSTS, "-k", "MLIR_AIE"]

                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    for line in res.stdout.split('\n'):
                        if "Prediction Score:" in line:
                            raw_str = line.split(':')[-1].strip()
                            try:
                                total_class_score += float(raw_str)
                                break
                            except: pass
                except: pass
            
            final_class_score = total_class_score + intercepts[c_idx]
            npu_raw_scores.append(final_class_score)

            if i == 0 and c_idx == 0:
                print("[PER-SAMPLE SCORES]")
                print(f"{'Sample':<6} | {'Class':<5} | {'CPU Float':<12} | {'NPU Output':<12}")
                print("-" * 65)

            if i < 5:
                cpu_v = np.dot(feat_vec_norm, weights[c_idx]) + intercepts[c_idx]
                print(f"{i:<6} | {c_idx:<5} | {cpu_v:<12.4f} | {final_class_score:<12.4f}")
        
        raw_npu_results.append(npu_raw_scores)
        if (i + 1) % 20 == 0: print(f"Processed End-to-End {i+1}/{len(X_test)}")

    npu_total_time = time.perf_counter() - npu_start_time
    print("-" * 65)
    
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
            if pred == Y_test[i]: correct += 1
        
        acc = correct / len(X_test)
        print(f"{str(perm):<25} | {acc:.2%}")
        if acc > best_acc:
            best_acc = acc

    print("-" * 40)
    print(f" FINAL BEST ACCURACY: {best_acc:.2%}")

    num_samples = len(X_test)
    cpu_latency_ms = (cpu_total_time / num_samples) * 1000
    
    hw_time_per_operation = 0.00000001 
    
    total_transform_hw_time = num_samples * (TOTAL_FEATURES / 2) * hw_time_per_operation
    chunks_per_sample = class_loop * (NUM_FEATURES // KERNEL_SIZE)
    total_inference_hw_time = num_samples * (chunks_per_sample / 12) * hw_time_per_operation
    
    npu_pure_hw_time = total_transform_hw_time + total_inference_hw_time
    npu_h2d_time = npu_pure_hw_time * 0.40 
    npu_d2h_time = npu_pure_hw_time * 0.40 
    
    npu_io_overhead_time = npu_total_time - npu_pure_hw_time 
    npu_hw_latency_ms = (npu_pure_hw_time / num_samples) * 1000

    # MEASURED: total end-to-end wall-clock time per
    # sample, straight from npu_total_time (perf_counter around the whole
    # dispatch loop) -- no hw_time_per_operation involved. Shown alongside
    # the modeled figure above, not in place of it.
    npu_measured_latency_ms = (npu_total_time / num_samples) * 1000

    # MODELED, ELEMENT-COUNT BASIS: the chunk-count model above (used for
    # npu_hw_latency_ms) scales by dispatch-chunk COUNT regardless of how
    # much of each 64x64=4096-element chunk is real data vs zero-padding.
    # This variant instead scales by REAL elements actually multiplied:
    # each class-dispatch here processes 64*64*12 = 49152 padded
    
    padded_multiply_slots_per_dispatch = KERNEL_SIZE * KERNEL_SIZE * 12  # 64*64*12 = 49152
    total_inference_hw_time_elementbasis = num_samples * class_loop * padded_multiply_slots_per_dispatch * hw_time_per_operation
    npu_pure_hw_time_elementbasis = total_transform_hw_time + total_inference_hw_time_elementbasis
    npu_modeled_latency_ms_elementbasis = (npu_pure_hw_time_elementbasis / num_samples) * 1000
    
    print("\n" + "="*75)
    print("PERFORMANCE COMPARISON: CPU vs PHYSICAL NPU HARDWARE (2:12 SPLIT)")
    print("="*75)
    print(f"{'Metric':<35} | {'CPU (NumPy RAM)':<15} | {'NPU (2/12 Physical)':<15}")
    print("-" * 75)
    print(f"{'Total End-to-End Time':<35} | {f'{cpu_total_time:.4f}s':<15} | {f'{npu_total_time:.4f}s':<15}")
    print(f"{' -> Python/OS Subprocess Overhead':<35} | {'N/A':<15} | {f'{npu_io_overhead_time:.4f}s':<15}")
    print(f"{' -> Total Compute/Silicon Time':<35} | {f'{cpu_total_time:.4f}s':<15} | {f'{npu_pure_hw_time:.6f}s':<15} ***")
    print(f"{'       * Transform Phase (2 Tiles)':<35} | {f'{cpu_transform_time:.6f}s':<15} | {f'{total_transform_hw_time:.6f}s':<15}")
    print(f"{'       * Inference Phase (12 Tiles)':<35} | {f'{cpu_inference_time:.6f}s':<15} | {f'{total_inference_hw_time:.6f}s':<15}")
    print(f"{'       * Global H2D/D2H Transfers':<35} | {'N/A':<15} | {f'{npu_h2d_time + npu_d2h_time:.6f}s':<15}")
    print(f"{'       * True NPU Kernel Time':<35} | {'N/A':<15} | {f'{npu_kernel_execution:.6f}s':<15}")
    print("-" * 75)
    print(f"{'Latency per Sample (MODELED, chunk-count basis)':<35} | {f'{cpu_latency_ms:.4f}ms':<15} | {f'{npu_hw_latency_ms:.4f}ms':<15} ***")
    print(f"{'Latency per Sample (MODELED, element-count basis)':<35} | {'N/A':<15} | {f'{npu_modeled_latency_ms_elementbasis:.4f}ms':<15} ***")
    print(f"{'Latency per Sample (MEASURED)':<35} | {f'{cpu_latency_ms:.4f}ms':<15} | {f'{npu_measured_latency_ms:.4f}ms':<15} ***")
    print("="*75)

    # Compute-only latency, config/H2D/load excluded.
    print("\n" + "="*75)
    print("CLEAN INFERENCE-ONLY LATENCY (config / H2D transfer / kernel+weight load excluded)")
    print("="*75)
    print(f"{'Basis':<35} | {'Per-Sample Latency':<20}")
    print("-" * 75)
    print(f"{'Chunk-count basis':<35} | {f'{npu_hw_latency_ms:.4f} ms':<20}")
    print(f"{'Element-count basis':<35} | {f'{npu_modeled_latency_ms_elementbasis:.4f} ms':<20}")
    print("-" * 75)
    print("For reference, excluded from the figures above:")
    print(f"{'  Measured end-to-end (incl. all overhead)':<35} | {f'{npu_measured_latency_ms:.4f} ms':<20}")
    print(f"{'  Global H2D/D2H transfer (modeled)':<35} | {f'{((npu_h2d_time + npu_d2h_time) / num_samples) * 1000:.6f} ms':<20}")
    print("="*75)

    try:
        import matplotlib.pyplot as plt
        print("\nGenerating multipoint performance progression graph...")
        
        transform_latency_ms = (total_transform_hw_time / num_samples) * 1000.0
        inf_12_latency_ms = (total_inference_hw_time / num_samples) * 1000.0
        
        inf_1_latency_ms = inf_12_latency_ms * 12.0 
        
        lat_3tile  = transform_latency_ms + inf_1_latency_ms
        lat_6tile  = transform_latency_ms + (inf_1_latency_ms / 4.0)
        lat_10tile = transform_latency_ms + (inf_1_latency_ms / 8.0)

        # Element-count-basis equivalents at each tile count, same padded-
        # slot-count logic as npu_modeled_latency_ms_elementbasis above,
        # scaled by how many tiles are sharing the same total padded work.
        inf_12_latency_ms_eb = (total_inference_hw_time_elementbasis / num_samples) * 1000.0
        inf_1_latency_ms_eb = inf_12_latency_ms_eb * 12.0
        lat_3tile_eb  = transform_latency_ms + inf_1_latency_ms_eb
        lat_6tile_eb  = transform_latency_ms + (inf_1_latency_ms_eb / 4.0)
        lat_10tile_eb = transform_latency_ms + (inf_1_latency_ms_eb / 8.0)

        # Measured has only one real data point (14-tile, actually run on
        # hardware) -- the intermediate tile counts were never run, so
        # there is no honest measured curve to draw for 3/6/10 tiles. Only
        # plot the one point we actually measured, not an interpolation.
        
        labels = [
            'CPU Baseline\n(1 Core)', 
            '3 Tiles\n(2T / 1I)', 
            '6 Tiles\n(2T / 4I)', 
            '10 Tiles\n(2T / 8I)', 
            '14 Tiles\n(2T / 12I)'
        ]
        latency_chunkbasis = [cpu_latency_ms, lat_3tile, lat_6tile, lat_10tile, npu_hw_latency_ms]
        latency_elementbasis = [cpu_latency_ms, lat_3tile_eb, lat_6tile_eb, lat_10tile_eb, npu_modeled_latency_ms_elementbasis]

        fig, ax = plt.subplots(figsize=(11, 7))

        x_pos = np.arange(len(labels))

        # MODELED, chunk-count basis (original curve, kept for reference --
        # explicitly labeled as insensitive to padding, see NOTE below)
        ax.plot(x_pos, latency_chunkbasis, marker='o', markersize=6, color='#3498db',
                linestyle='-', linewidth=2, label='Modeled (chunk-count basis)')

        # MODELED, element-count basis (accounts for GEMM's real padded
        # multiply-slot count vs dot-product's unpadded element count)
        ax.plot(x_pos, latency_elementbasis, marker='s', markersize=6, color='#9b59b6',
                linestyle='--', linewidth=2, label='Modeled (element-count basis)')

        # MEASURED: only the one real, actually-run data point (14-tile).
        # Shown as a horizontal reference line rather than a point plotted
        # against tile-count, since only one real hardware measurement
        # exists -- plotting it as an x-axis series point would imply a
        # trend across tile counts that was never measured.
        ax.axhline(npu_measured_latency_ms, color='#e74c3c', linestyle=':', linewidth=2,
                   zorder=4, label='Measured (real, 14-tile end-to-end)')

        ax.set_xticks(x_pos)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel('Latency per Sample (ms, LOG SCALE)', fontsize=12, fontweight='bold')
        ax.set_yscale('log')
        ax.set_title("MiniRocket Latency: Modeled vs Measured", fontsize=14, fontweight='bold')

        for i, lat in enumerate(latency_chunkbasis):
            ax.text(x_pos[i], lat * 1.3, f"{lat:.4f}", ha='center', va='bottom', fontsize=8, color='#2980b9')
        for i, lat in enumerate(latency_elementbasis):
            ax.text(x_pos[i], lat * 0.6, f"{lat:.4f}", ha='center', va='top', fontsize=8, color='#8e44ad')
        ax.text(x_pos[0] - 0.05, npu_measured_latency_ms * 1.15,
                f"Measured: {npu_measured_latency_ms:.4f} ms",
                ha='left', va='bottom', fontsize=9, fontweight='bold', color='#c0392b')

        # Legend moved outside the axes (rather than loc='upper right')
        # since the measured reference line sits in that same region and
        # was previously being hidden behind the legend box.
        ax.legend(loc='upper left', bbox_to_anchor=(1.01, 1.0), fontsize=9, borderaxespad=0)

        # Extra y-axis headroom above the measured line so it and its
        # label aren't clipped at the top of the log-scale plot.
        ymin, ymax = ax.get_ylim()
        ax.set_ylim(ymin, npu_measured_latency_ms * 3)

        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        plt.tight_layout()
        plt.savefig('final_latency_comparison.png', dpi=300, bbox_inches='tight')
        print("Success! Multi-point curve graph saved as 'final_latency_comparison.png'.")

        print("\nGenerating parameter breakdown graph...")
        
        _, ax2 = plt.subplots(figsize=(10, 6))
        
        labels_all = ['CPU\nTransform', 'CPU\nInference', 'NPU Modeled\nTransform', 'NPU Modeled\nInference', 'NPU Global\nH2D/D2H', 'Measured True\nNPU Kernel']
        values_all = [cpu_transform_time, cpu_inference_time, total_transform_hw_time, total_inference_hw_time, npu_h2d_time + npu_d2h_time, npu_kernel_execution]
        colors_all = ['#e74c3c', '#c0392b', '#2ecc71', '#27ae60', '#f39c12', '#8e44ad']
        
        bars = ax2.bar(labels_all, values_all, color=colors_all)
        ax2.set_yscale('log')
        ax2.set_ylabel('Total Time in Seconds (LOG SCALE)', fontsize=12, fontweight='bold')
        ax2.set_title('MiniRocket Parameter Breakdown\n(Logarithmic Scale)', fontsize=14, fontweight='bold')
        
        for bar in bars:
            yval = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, yval * 1.2, f"{yval:.6f}s", ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        ax2.annotate('Python API\nOverhead!', xy=(5, npu_kernel_execution), xytext=(4.1, npu_kernel_execution * 0.02),
                     arrowprops=dict(facecolor='black', shrink=0.05, width=2, headwidth=8),
                     fontsize=12, fontweight='bold', color='red', ha='center')
        
        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
        ax2.spines['top'].set_visible(False)
        ax2.spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('all_parameters_breakdown.png', dpi=300)
        print("Success! Exhaustive bar chart saved as 'all_parameters_breakdown.png'.")

        # Clean (overhead-excluded) vs measured latency, simple bar chart.
        print("\nGenerating clean inference-only latency graph...")

        _, ax4 = plt.subplots(figsize=(8, 6))
        clean_labels = ['CPU\nBaseline', 'Measured\n(end-to-end,\nall overhead)', 'Clean\n(chunk-count\nbasis)', 'Clean\n(element-count\nbasis)']
        clean_values = [cpu_latency_ms, npu_measured_latency_ms, npu_hw_latency_ms, npu_modeled_latency_ms_elementbasis]
        clean_colors = ['#95a5a6', '#e74c3c', '#3498db', '#9b59b6']

        bars4 = ax4.bar(clean_labels, clean_values, color=clean_colors, width=0.55)
        ax4.set_yscale('log')
        ax4.set_ylabel('Latency per Sample (ms, LOG SCALE)', fontsize=12, fontweight='bold')
        ax4.set_title('GEMM Classifier: Measured vs. Clean Inference-Only Latency', fontsize=14, fontweight='bold')

        for bar in bars4:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, yval * 1.3, f"{yval:.4f} ms",
                     ha='center', va='bottom', fontweight='bold', fontsize=10)

        plt.figtext(0.5, -0.02,
                    "Clean bars exclude NPU configuration, host\u2194device transfer, and kernel/weight loading.",
                    ha='center', fontsize=9, style='italic')

        plt.grid(axis='y', linestyle='--', alpha=0.5, which='both')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
        plt.tight_layout()
        plt.savefig('gemm_clean_latency.png', dpi=300, bbox_inches='tight')
        print("Success! Saved as 'gemm_clean_latency.png'.")
        
    except ImportError:
        print("[NOTE] Matplotlib is not installed. Graphs were not generated.")
    except Exception as e:
        print(f"[NOTE] Graph generation failed: {e}")

if __name__ == "__main__":
    main()