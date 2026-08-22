#!/usr/bin/env python3
"""
minirocket_transform_kernel.py

STATUS: new for Step 1 of on-chip-execution fusion work. Not yet run
through aiecc.py, not compiled, not run on hardware.

Generates MLIR for the REAL MiniROCKET transform (dilation + 84
kernel-index-triples + padding-alternated PPV + normalization),
replacing the contiguous-window stand-in previously used in
npu_end_to_end.py / dotproduct_minirocket.py's shared transform stage.

Built as a direct structural mirror of the project's own proven pattern
(dotproduct12_v3_kernel.py: plain Python string templating emitting MLIR
text directly, NOT the aie.dialects.aie/mlir_mod_ctx Python DSL used in
minirocket_kernel.py/whole_array.py -- chosen to match the ONE generator
in this project confirmed structurally consistent with a real,
hardware-validated xclbin's connectivity dump).

WHAT'S THE SAME AS THE EXISTING (CONTIGUOUS-WINDOW) TRANSFORM:
  - 2 compute tiles (UNCHANGED tile count, per project standing rule)
  - Device npu1
  - shim -> mem -> compute per-tile staging

WHAT'S DIFFERENT (the actual new work):
  - Per-feature dilation/kernel-index/padding math (real MiniROCKET,
    not a contiguous 9-tap window)
  - PPV pooling and normalization now happen ON-CHIP (previously CPU-side)
  - One call to minirocket_feature_transform() per feature (420 calls
    per tile, Path A -- see FUSION_PLAN.md Step 1e), rather than one
    call per sliding-window position

Feature schedule (which dilation/kernel_index/padding each of the 840
output features corresponds to) is precomputed here in Python, at
generation time -- matches build_schedule.py's logic exactly (see that
file for the standalone, independently-verified version: 840 entries,
correct per-dilation counts, all padding values valid for SEQ_LENGTH=251).
"""

from itertools import combinations

SEQ_LENGTH = 251
DILATIONS = [1, 2, 3, 4, 6, 9, 14, 21, 31]
NUM_FEATURES_PER_DILATION = [2, 1, 1, 1, 1, 1, 1, 1, 1]
NUM_KERNEL_INDICES = 84
KERNEL_TAPS = 9
TOTAL_FEATURES = 840
FEATURES_PER_TILE = 420  # 2-tile split, matches existing transform's FEATURES_PER_CORE
DEVICE_SYMBOL = "npu1"

KERNEL_INDICES = list(combinations(range(KERNEL_TAPS), 3))
assert len(KERNEL_INDICES) == NUM_KERNEL_INDICES


def build_feature_schedule():
    """Matches minirocket.py transform()'s iteration order exactly.
    Independently verified in build_schedule.py before this generator
    was written -- see that file's output for the correctness checks
    (840 total, per-dilation counts, padding validity) this logic must
    continue to satisfy."""
    schedule = []
    for dilation_index, dilation in enumerate(DILATIONS):
        padding0 = dilation_index % 2
        n_feat_this_dilation = NUM_FEATURES_PER_DILATION[dilation_index]
        padding = ((KERNEL_TAPS - 1) * dilation) // 2
        for kernel_index in range(NUM_KERNEL_INDICES):
            padding1 = (padding0 + kernel_index) % 2
            i0, i1, i2 = KERNEL_INDICES[kernel_index]
            for _ in range(n_feat_this_dilation):
                schedule.append({
                    "dilation": dilation,
                    "padding": padding,
                    "kernel_i0": i0,
                    "kernel_i1": i1,
                    "kernel_i2": i2,
                    "padding1": padding1,
                })
    assert len(schedule) == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {len(schedule)}"
    return schedule


def generate_mlir():
    schedule = build_feature_schedule()
    tile_schedules = [schedule[:FEATURES_PER_TILE], schedule[FEATURES_PER_TILE:]]

    lines = []
    w = lines.append

    w("module {")
    w(f"  aie.device({DEVICE_SYMBOL}) {{")
    w(f"    func.func private @zero_i32_scalar(memref<1xi32>)")
    w(
        f"    func.func private @minirocket_feature_transform("
        f"memref<{SEQ_LENGTH}xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<1xi32>)"
    )
    w("")

    # --- Tiles: 2 columns (0, 1), one compute tile each (row 2) ---
    # NOTE: shim(row0)/mem(row1)/compute(row2) per column, matching the
    # classifier's per-column staging pattern but with only 1 compute
    # tile per column here (vs. classifier's 4), since transform stays
    # at 2 total compute tiles.
    for col in range(2):
        w(f"    %shim_noc_tile_{col}_0 = aie.tile({col}, 0)")
        w(f"    %mem_tile_{col}_1 = aie.tile({col}, 1)")
        w(f"    %tile_{col}_2 = aie.tile({col}, 2)")
    w("")

    # --- Routing: per column, series/bias/scaler_mean/scaler_scale in,
    # normalized-feature out ---
    for col in range(2):
        w(f"    // --- ROUTING: COLUMN {col} (transform tile {col}) ---")
        w(f"    aie.objectfifo @series_in_{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{SEQ_LENGTH}xi32>>")
        w(f"    aie.objectfifo @series_mem_{col}(%mem_tile_{col}_1, {{%tile_{col}_2}}, 2 : i32) : !aie.objectfifo<memref<{SEQ_LENGTH}xi32>>")
        w(f"    aie.objectfifo.link [@series_in_{col}] -> [@series_mem_{col}]([] [])")

        w(f"    aie.objectfifo @bias_in_{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo @bias_mem_{col}(%mem_tile_{col}_1, {{%tile_{col}_2}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo.link [@bias_in_{col}] -> [@bias_mem_{col}]([] [])")

        w(f"    aie.objectfifo @smean_in_{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo @smean_mem_{col}(%mem_tile_{col}_1, {{%tile_{col}_2}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo.link [@smean_in_{col}] -> [@smean_mem_{col}]([] [])")

        w(f"    aie.objectfifo @sscale_in_{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo @sscale_mem_{col}(%mem_tile_{col}_1, {{%tile_{col}_2}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo.link [@sscale_in_{col}] -> [@sscale_mem_{col}]([] [])")

        w(f"    aie.objectfifo @out_mem_{col}(%tile_{col}_2, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo @out_{col}(%mem_tile_{col}_1, {{%shim_noc_tile_{col}_0}}, 2 : i32) : !aie.objectfifo<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo.link [@out_mem_{col}] -> [@out_{col}]([] [])")
        w("")

    # --- Cores: one per column, body computes all FEATURES_PER_TILE
    # features for its tile, one minirocket_feature_transform() call
    # per feature, each with compile-time-constant dilation/padding/
    # kernel-index args baked in via this Python loop (Path A: no
    # cross-feature caching, see FUSION_PLAN.md Step 1e) ---
    w("    // --- CORES ---")
    for col in range(2):
        tile_schedule = tile_schedules[col]
        w(f"    %core_{col}_2 = aie.core(%tile_{col}_2) {{")
        w("      %c0 = arith.constant 0 : index")
        w("      %c4294967295 = arith.constant 4294967295 : index")
        w("      %c1 = arith.constant 1 : index")
        w("      scf.for %arg0 = %c0 to %c4294967295 step %c1 {")
        w(f"        %series = aie.objectfifo.acquire @series_mem_{col}(Consume, 1) : !aie.objectfifosubview<memref<{SEQ_LENGTH}xi32>>")
        w(f"        %series_v = aie.objectfifo.subview.access %series[0] : !aie.objectfifosubview<memref<{SEQ_LENGTH}xi32>> -> memref<{SEQ_LENGTH}xi32>")
        w(f"        %bias = aie.objectfifo.acquire @bias_mem_{col}(Consume, 1) : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"        %bias_v = aie.objectfifo.subview.access %bias[0] : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>> -> memref<{FEATURES_PER_TILE}xi32>")
        w(f"        %smean = aie.objectfifo.acquire @smean_mem_{col}(Consume, 1) : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"        %smean_v = aie.objectfifo.subview.access %smean[0] : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>> -> memref<{FEATURES_PER_TILE}xi32>")
        w(f"        %sscale = aie.objectfifo.acquire @sscale_mem_{col}(Consume, 1) : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"        %sscale_v = aie.objectfifo.subview.access %sscale[0] : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>> -> memref<{FEATURES_PER_TILE}xi32>")
        w(f"        %out = aie.objectfifo.acquire @out_mem_{col}(Produce, 1) : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>>")
        w(f"        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<{FEATURES_PER_TILE}xi32>> -> memref<{FEATURES_PER_TILE}xi32>")
        w("")

        for feat_idx, entry in enumerate(tile_schedule):
            dilation = entry["dilation"]
            padding = entry["padding"]
            i0, i1, i2 = entry["kernel_i0"], entry["kernel_i1"], entry["kernel_i2"]
            padding1 = entry["padding1"]

            w(f"        // feature {feat_idx}: dilation={dilation} padding={padding} kernel=({i0},{i1},{i2}) padding1={padding1}")
            w(f"        %fidx_{feat_idx} = arith.constant {feat_idx} : index")
            w(f"        %bias_{feat_idx} = memref.load %bias_v[%fidx_{feat_idx}] : memref<{FEATURES_PER_TILE}xi32>")
            w(f"        %smean_{feat_idx} = memref.load %smean_v[%fidx_{feat_idx}] : memref<{FEATURES_PER_TILE}xi32>")
            w(f"        %sscale_{feat_idx} = memref.load %sscale_v[%fidx_{feat_idx}] : memref<{FEATURES_PER_TILE}xi32>")
            w(f"        %out_slot_{feat_idx} = memref.subview %out_v[%fidx_{feat_idx}] [1] [1] : memref<{FEATURES_PER_TILE}xi32> to memref<1xi32, strided<[1], offset: ?>>")
            w(f"        %dilation_{feat_idx} = arith.constant {dilation} : i32")
            w(f"        %padding_{feat_idx} = arith.constant {padding} : i32")
            w(f"        %ki0_{feat_idx} = arith.constant {i0} : i32")
            w(f"        %ki1_{feat_idx} = arith.constant {i1} : i32")
            w(f"        %ki2_{feat_idx} = arith.constant {i2} : i32")
            w(f"        %pad1_{feat_idx} = arith.constant {padding1} : i32")
            w(
                f"        func.call @minirocket_feature_transform(%series_v, %bias_{feat_idx}, "
                f"%smean_{feat_idx}, %sscale_{feat_idx}, %dilation_{feat_idx}, %padding_{feat_idx}, "
                f"%ki0_{feat_idx}, %ki1_{feat_idx}, %ki2_{feat_idx}, %pad1_{feat_idx}, %out_slot_{feat_idx}) : "
                f"(memref<{SEQ_LENGTH}xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, "
                f"memref<1xi32, strided<[1], offset: ?>>) -> ()"
            )
            w("")

        w(f"        aie.objectfifo.release @series_mem_{col}(Consume, 1)")
        w(f"        aie.objectfifo.release @bias_mem_{col}(Consume, 1)")
        w(f"        aie.objectfifo.release @smean_mem_{col}(Consume, 1)")
        w(f"        aie.objectfifo.release @sscale_mem_{col}(Consume, 1)")
        w(f"        aie.objectfifo.release @out_mem_{col}(Produce, 1)")
        w("      }")
        w("      aie.end")
        w("    } {stack_size = 4096 : i32}")
    w("")

    # --- Runtime sequence ---
    w("    // --- HOST RUNTIME SEQUENCE ---")
    w(
        f"    aie.runtime_sequence(%arg0: memref<{SEQ_LENGTH * 2}xi32>, "
        f"%arg1: memref<{TOTAL_FEATURES}xi32>, %arg2: memref<{TOTAL_FEATURES}xi32>, "
        f"%arg3: memref<{TOTAL_FEATURES}xi32>, %arg4: memref<{TOTAL_FEATURES}xi32>) {{"
    )
    dma_id = 0
    for col in range(2):
        series_off = col * SEQ_LENGTH
        feat_off = col * FEATURES_PER_TILE
        w(f"      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, {series_off}][1, 1, 1, {SEQ_LENGTH}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @series_in_{col}}} : memref<{SEQ_LENGTH * 2}xi32>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, {feat_off}][1, 1, 1, {FEATURES_PER_TILE}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @bias_in_{col}}} : memref<{TOTAL_FEATURES}xi32>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, {feat_off}][1, 1, 1, {FEATURES_PER_TILE}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @smean_in_{col}}} : memref<{TOTAL_FEATURES}xi32>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg3[0, 0, 0, {feat_off}][1, 1, 1, {FEATURES_PER_TILE}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @sscale_in_{col}}} : memref<{TOTAL_FEATURES}xi32>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg4[0, 0, 0, {feat_off}][1, 1, 1, {FEATURES_PER_TILE}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @out_{col}}} : memref<{TOTAL_FEATURES}xi32>")
        dma_id += 1
    for col in range(2):
        w(f"      aiex.npu.dma_wait {{symbol = @out_{col}}}")
    w("    }")
    w("  }")
    w("}")

    return "\n".join(lines)


if __name__ == "__main__":
    mlir_text = generate_mlir()
    with open("minirocket_transform_v1.mlir", "w") as f:
        f.write(mlir_text)
    n_lines = mlir_text.count("\n")
    print(f"Wrote minirocket_transform_v1.mlir ({n_lines} lines)")
    print(f"2 columns x 1 compute tile = 2 compute tiles, {FEATURES_PER_TILE} features/tile, "
          f"{TOTAL_FEATURES} total features")
    print("Path A: one minirocket_feature_transform() call per feature, no cross-feature caching")
