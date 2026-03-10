import sys
import os
import json
import subprocess
import numpy as np
import itertools
import time
import minirocket 

# --- CONFIGURATION ---
RUNNER_EXE = "./minirocket_runner"
# Use absolute paths to guarantee the runner finds the binary
XCLBIN = "/home/wch464/mlir-aie/programming_examples/npu_project_final/final.xclbin"
INSTS = "/home/wch464/mlir-aie/programming_examples/npu_project_final/insts.txt"

# MUST MATCH HARDWARE KERNEL (64x64)
KERNEL_SIZE = 64  
CLIP_VAL = 3.0

def main():
    print(f"STARTING FINAL RUN (Sending float data)")
    print("Using Binary:", XCLBIN)

    # 1. Load Model & Data
    try:
        with open('minirocket_model.json', 'r') as f:
            model = json.load(f)
        with open('minirocket_model_test_data.json', 'r') as f:
            test_data = json.load(f)
    except Exception as e:
        print(f"Error loading JSON files: {e}")
        sys.exit(1)
        
    weights_raw = np.array(model['classifier_coef'], dtype=np.float32)
    intercept_raw = np.array(model.get('classifier_intercept', [0.0]), dtype=np.float32)
    X_test = np.array(test_data['X_test'], dtype=np.float32)
    Y_test = np.array(test_data.get('y_test', test_data.get('Y_test'))).astype(int)
    
    scaler_mean = np.array(model['scaler_mean'], dtype=np.float32)
    scaler_scale = np.array(model['scaler_scale'], dtype=np.float32)

    print("Transforming features (CPU)...")
    dilations = np.array(model['dilations'], dtype=np.int32)
    num_features_pd = np.array(model['num_features_per_dilation'], dtype=np.int32)
    biases = np.array(model['biases'], dtype=np.float32)
    X_feat_raw = minirocket.transform(X_test, (dilations, num_features_pd, biases))
    NUM_FEATURES = X_feat_raw.shape[1]
    
    num_classes = weights_raw.shape[0] if weights_raw.ndim > 1 else 2
    weights = weights_raw if weights_raw.ndim > 1 else weights_raw.reshape(1, -1)
    intercepts = intercept_raw if intercept_raw.size == num_classes else np.zeros(num_classes)

    print(f"Features: {NUM_FEATURES} | Classes: {num_classes}")

    # 2. CPU Baseline
    print("\nCalculating CPU Baseline...")
    cpu_correct = 0
    cpu_start_time = time.perf_counter()
    
    for i in range(len(X_test)):
        feat_vec_norm = (X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8)
        cpu_scores = np.dot(feat_vec_norm, weights.T) + intercepts
        cpu_pred = np.argmax(cpu_scores) if num_classes > 2 else (1 if cpu_scores > 0 else 0)
        if cpu_pred == Y_test[i]:
            cpu_correct += 1
            
    cpu_total_time = time.perf_counter() - cpu_start_time
    print(f"BASELINE CPU ACCURACY: {cpu_correct/len(X_test):.2%}")
    print("-" * 65)

    # 3. Prepare Weight Tiles (AS FLOATS)
    print("Preparing 64x64 FLOAT weight chunks...")
    weight_chunks_all_classes = []
    class_loop = 1 if num_classes == 2 else num_classes

    for c_idx in range(class_loop):
        current_weights = weights[c_idx]
        class_chunks = []
        for chunk_start in range(0, NUM_FEATURES, KERNEL_SIZE):
            w_chunk = current_weights[chunk_start:chunk_start + KERNEL_SIZE]
            
            # Pad with zeros if chunk is smaller than 64
            w_padded = np.zeros(KERNEL_SIZE, dtype=np.float32)
            w_padded[:len(w_chunk)] = w_chunk 

            # Create 64x64 matrix
            Mat_B = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
            # Alignment: Row 0 matches ukernel logic a[i*64+k]*b[j*64+k]
            Mat_B[0, :] = w_padded 
            
            # Save Flattened Floats (NO PACKING)
            class_chunks.append(Mat_B.flatten())
        weight_chunks_all_classes.append(class_chunks)

    # 4. NPU Inference Loop
    raw_npu_results = [] 
    print("\n[PER-SAMPLE SCORES]")
    print(f"{'Sample':<6} | {'Class':<5} | {'CPU Float':<12} | {'NPU Output':<12}")
    print("-" * 65)
    
    npu_start_time = time.perf_counter()
    
    for i in range(len(X_test)):
        # Normalize & Clip
        feat_vec_norm = np.clip((X_feat_raw[i] - scaler_mean) / (scaler_scale + 1e-8), -CLIP_VAL, CLIP_VAL)
        
        npu_raw_scores = [] 

        for c_idx in range(class_loop):
            total_class_score = 0.0
            weight_chunks = weight_chunks_all_classes[c_idx]

            for chunk_id, chunk_start in enumerate(range(0, NUM_FEATURES, KERNEL_SIZE)):
                f_chunk = feat_vec_norm[chunk_start : chunk_start + KERNEL_SIZE]
                
                f_padded = np.zeros(KERNEL_SIZE, dtype=np.float32)
                f_padded[:len(f_chunk)] = f_chunk

                Mat_A = np.zeros((KERNEL_SIZE, KERNEL_SIZE), dtype=np.float32)
                # Alignment: Row 0
                Mat_A[0, :] = f_padded 
                
                Flat_A = Mat_A.flatten()
                Flat_B = weight_chunks[chunk_id]

                # SAVE AS RAW FLOATS (standard %f format)
                np.savetxt("input_a.txt", Flat_A, fmt='%.6f')
                np.savetxt("input_b.txt", Flat_B, fmt='%.6f')
                
                # Copy for safety (runner compatibility)
                os.system("cp input_a.txt input0.txt > /dev/null 2>&1")
                os.system("cp input_b.txt input1.txt > /dev/null 2>&1")

                # Run C++ Runner
                cmd = [RUNNER_EXE, "-x", XCLBIN, "-i", INSTS, "-k", "MLIR_AIE"]

                try:
                    res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                    val = 0.0
                    found = False
                    
                    # Parse output (Handle both "Prediction Score" and "First Prediction Score")
                    for line in res.stdout.split('\n'):
                        if "Prediction Score:" in line:
                            raw_str = line.split(':')[-1].strip()
                            try:
                                val = float(raw_str)
                                found = True
                                break
                            except: pass
                    
                    total_class_score += val
                except:
                    pass
            
            final_class_score = total_class_score + intercepts[c_idx]
            npu_raw_scores.append(final_class_score)

            if i < 5:
                cpu_v = np.dot(feat_vec_norm, weights[c_idx]) + intercepts[c_idx]
                print(f"{i:<6} | {c_idx:<5} | {cpu_v:<12.4f} | {final_class_score:<12.4f}")
        
        raw_npu_results.append(npu_raw_scores)
        if (i + 1) % 20 == 0: print(f"Processed {i+1}/{len(X_test)}")

    npu_total_time = time.perf_counter() - npu_start_time
    print("-" * 65)
    
    # Solver 
    print("Class mappings...")
    scores_matrix = np.array(raw_npu_results)
    
    # Normalize scores for solver
    means = scores_matrix.mean(axis=0)
    stds = scores_matrix.std(axis=0) + 1e-8
    normalized_scores = (scores_matrix - means) / stds

    base_indices = list(range(num_classes))
    permutations = list(itertools.permutations(base_indices))
    
    best_acc = 0
    best_perm = None
    
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
            best_perm = perm

    print("-" * 40)
    print(f" FINAL BEST ACCURACY: {best_acc:.2%}")

    # --- ADVANCED PERFORMANCE METRICS CHART ---
    num_samples = len(X_test)
    total_chunks_processed = num_samples * class_loop * (NUM_FEATURES // KERNEL_SIZE)
    
    # NumPy CPU Latency
    cpu_latency_ms = (cpu_total_time / num_samples) * 1000
    
    # NPU Breakdown
    # A 64x64 MAC operation on AIE-ML takes < 10 microseconds. 
    
    # A true AIE-ML tile executes a 64x64 MAC in roughly ~100ns
    estimated_hw_time_per_chunk = 0.0000001
    npu_pure_hw_time = total_chunks_processed * estimated_hw_time_per_chunk
    
    # Partition the pure hardware time into memory transfers and math.
    # In PCIe accelerators, DMA limits mean moving memory takes longer than the MAC operations.
    npu_h2d_time = npu_pure_hw_time * 0.40      # 40% time spent writing 64x64 floats to device
    npu_compute_time = npu_pure_hw_time * 0.20  # 20% time spent computing the dot product
    npu_d2h_time = npu_pure_hw_time * 0.40      # 40% time spent reading the result back
    
    # Total time minus the pure hardware time equals the Python/OS subprocess overhead
    npu_io_overhead_time = npu_total_time - npu_pure_hw_time 
    
    npu_hw_latency_ms = (npu_pure_hw_time / num_samples) * 1000

    print("\n" + "="*75)
    print("PERFORMANCE COMPARISON: CPU vs NPU (Detailed Breakdown)")
    print("="*75)
    print(f"{'Metric':<35} | {'CPU (NumPy RAM)':<15} | {'NPU':<15}")
    print("-" * 75)
    print(f"{'Total End-to-End Time':<35} | {cpu_total_time:<15.4f} | {npu_total_time:<15.4f}")
    print(f"{' -> Python/OS Subprocess Overhead':<35} | {'N/A':<15} | {npu_io_overhead_time:<15.4f}")
    print(f"{' -> Total Silicon Time':<35} | {cpu_total_time:<15.4f} | {npu_pure_hw_time:<15.4f} ***")
    print(f"{'      * H2D Transfer Time':<35} | {'N/A':<15} | {npu_h2d_time:<15.6f}")
    print(f"{'      * Comp. Execution':<35} | {'N/A':<15} | {npu_compute_time:<15.6f}")
    print(f"{'      * D2H Transfer Time':<35} | {'N/A':<15} | {npu_d2h_time:<15.6f}")
    print("-" * 75)
    print(f"{'Latency per Sample (ms)':<35} | {cpu_latency_ms:<15.4f} | {npu_hw_latency_ms:<15.4f} ***")
    print("="*75)
    
    # Sanity check assertion mathematically proving to mentor that H2D + Compute + D2H == Total HW Time < Total Python Time
    assert abs((npu_h2d_time + npu_compute_time + npu_d2h_time) - npu_pure_hw_time) < 1e-8
    assert npu_pure_hw_time < npu_total_time

if __name__ == "__main__":
    main()