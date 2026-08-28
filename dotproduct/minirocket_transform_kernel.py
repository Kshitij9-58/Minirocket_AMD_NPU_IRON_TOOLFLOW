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

import argparse
import os
from itertools import combinations

SEQ_LENGTH = 251
DILATIONS = [1, 2, 3, 4, 6, 9, 14, 21, 31]
NUM_FEATURES_PER_DILATION = [2, 1, 1, 1, 1, 1, 1, 1, 1]
NUM_KERNEL_INDICES = 84
KERNEL_TAPS = 9
TOTAL_FEATURES = 840
FEATURES_PER_TILE = 420  # 2-tile split, matches existing transform's FEATURES_PER_CORE
DEVICE_SYMBOL = "npu1"

# TEST_FEATURE_LIMIT: if set (not None), each tile only processes this many
# features instead of the full FEATURES_PER_TILE (420). Used for small-scale
# validation builds (Step 3 gate) before committing to the full 420-feature
# unroll, which has proven too slow/unreliable to compile in two attempts
# tonight. Output buffer sizes and DMA transfer sizes shrink to match --
# this is a genuinely smaller kernel, not a truncated run of the full one,
# so it validates structure/correctness cheaply without being representative
# of full-scale compile time.
TEST_FEATURE_LIMIT = None
ACTIVE_FEATURES_PER_TILE = TEST_FEATURE_LIMIT if TEST_FEATURE_LIMIT is not None else FEATURES_PER_TILE

# Absolute path to the compiled .o file, resolved at GENERATION time from
# the current working directory. A bare filename ("minirocket_transform_kernel.o")
# was tried first and failed with "undefined symbol" across multiple
# real aiecc.py runs even when the .o file genuinely existed in the same
# directory as the .mlir -- xchesscc's link_with resolution appears to
# need either an absolute path or a path relative to wherever it's
# actually invoked from (which may be a nested work/ subdirectory, not
# the directory containing the .mlir). An absolute path removes the
# ambiguity entirely regardless of invocation directory. IMPORTANT: this
# means minirocket_transform_kernel.py MUST be run from the SAME
# directory where minirocket_transform_kernel.o will be compiled (i.e.
# run `python3 minirocket_transform_kernel.py` from inside
# transform_v1_build/ on wolverine, not from some other directory and
# copied over afterward) -- otherwise this path will point to a
# location that doesn't match where the .o actually ends up.
LINK_WITH_OBJ_PATH = os.path.join(os.getcwd(), "minirocket_transform_kernel.o")

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
    # Truncate each tile's schedule to ACTIVE_FEATURES_PER_TILE (8 for this
    # small-scale validation build, 420 for the real full design) -- slices
    # taken from the START of each tile's real 420-feature block, so the
    # dilation/kernel-index values tested here are genuine early entries
    # from the real schedule, not synthetic placeholders.
    tile_schedules = [
        schedule[:FEATURES_PER_TILE][:ACTIVE_FEATURES_PER_TILE],
        schedule[FEATURES_PER_TILE:][:ACTIVE_FEATURES_PER_TILE],
    ]

    # Combined input layout per tile: [series(SEQ_LENGTH) | bias(FEATURES_PER_TILE)
    # | smean(FEATURES_PER_TILE) | sscale(FEATURES_PER_TILE)], one flat buffer.
    # Consolidating 4 separate input ObjectFifos into 1 avoids exceeding the
    # shim tile's output DMA channel limit (confirmed via a real aiecc.py
    # failure -- "number of output DMA channel exceeded" -- when this kernel
    # used 4 input streams + 1 output stream per column; the proven
    # dotproduct12_v3.mlir classifier only ever uses 2 in + 1 out per column).
    COMBINED_IN_LEN = SEQ_LENGTH + 3 * ACTIVE_FEATURES_PER_TILE
    SERIES_OFFSET = 0
    BIAS_OFFSET = SEQ_LENGTH
    SMEAN_OFFSET = SEQ_LENGTH + ACTIVE_FEATURES_PER_TILE
    SSCALE_OFFSET = SEQ_LENGTH + 2 * ACTIVE_FEATURES_PER_TILE

    lines = []
    w = lines.append

    w("module {")
    w(f"  aie.device({DEVICE_SYMBOL}) {{")
    w(f"    func.func private @zero_i32_scalar(memref<1xi32>)")
    w(
        f'    func.func private @minirocket_feature_transform('
        f"memref<{COMBINED_IN_LEN}xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, "
        f'memref<{ACTIVE_FEATURES_PER_TILE}xi32>, i32)'
    )
    w("")

    # --- Tiles: 2 columns (0, 1), one compute tile each (row 2) ---
    for col in range(2):
        w(f"    %shim_noc_tile_{col}_0 = aie.tile({col}, 0)")
        w(f"    %mem_tile_{col}_1 = aie.tile({col}, 1)")
        w(f"    %tile_{col}_2 = aie.tile({col}, 2)")
    w("")

    # --- Routing: per column, ONE combined input stream, ONE output stream ---
    for col in range(2):
        w(f"    // --- ROUTING: COLUMN {col} (transform tile {col}) ---")
        w(f"    aie.objectfifo @in_{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{COMBINED_IN_LEN}xi32>>")
        w(f"    aie.objectfifo @in_mem_{col}(%mem_tile_{col}_1, {{%tile_{col}_2}}, 2 : i32) : !aie.objectfifo<memref<{COMBINED_IN_LEN}xi32>>")
        w(f"    aie.objectfifo.link [@in_{col}] -> [@in_mem_{col}]([] [])")

        w(f"    aie.objectfifo @out_mem_{col}(%tile_{col}_2, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{ACTIVE_FEATURES_PER_TILE}xi32>>")
        w(f"    aie.objectfifo @out_{col}(%mem_tile_{col}_1, {{%shim_noc_tile_{col}_0}}, 2 : i32) : !aie.objectfifo<memref<{ACTIVE_FEATURES_PER_TILE}xi32>>")
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
        w(f"      %c_series_off = arith.constant {SERIES_OFFSET} : i32")
        w("")

        # NO memref.alloc() anywhere in this core body -- confirmed via
        # real hardware failure ("could not find free space for
        # SpaceSymbol _ZL4pool in memory DMb") that memref.alloc() lowers
        # to a heap allocation requiring libc's malloc pool, and this
        # AIE-ML tile's 4 local memory banks (64KB total) are already
        # fully claimed by the toolchain's own default static
        # allocations, leaving no room for a heap pool. Confirmed by
        # comparing against the proven dotproduct12_v3.mlir classifier,
        # whose map files contain NO _ZL4pool/malloc symbols at all,
        # because it never calls memref.alloc() -- only reads/writes
        # ObjectFifo-provided buffers directly. This version does the
        # same: minirocket_feature_transform() now takes the FULL
        # combined input buffer + a series offset, and the FULL output
        # buffer + a target index, doing its own pointer arithmetic in
        # C++ rather than requiring MLIR-side scratch copies.
        w("      scf.for %arg0 = %c0 to %c4294967295 step %c1 {")
        w(f"        %in_elem = aie.objectfifo.acquire @in_mem_{col}(Consume, 1) : !aie.objectfifosubview<memref<{COMBINED_IN_LEN}xi32>>")
        w(f"        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<{COMBINED_IN_LEN}xi32>> -> memref<{COMBINED_IN_LEN}xi32>")
        w(f"        %out = aie.objectfifo.acquire @out_mem_{col}(Produce, 1) : !aie.objectfifosubview<memref<{ACTIVE_FEATURES_PER_TILE}xi32>>")
        w(f"        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<{ACTIVE_FEATURES_PER_TILE}xi32>> -> memref<{ACTIVE_FEATURES_PER_TILE}xi32>")
        w("")

        for feat_idx, entry in enumerate(tile_schedule):
            dilation = entry["dilation"]
            padding = entry["padding"]
            i0, i1, i2 = entry["kernel_i0"], entry["kernel_i1"], entry["kernel_i2"]
            padding1 = entry["padding1"]

            bias_idx = BIAS_OFFSET + feat_idx
            smean_idx = SMEAN_OFFSET + feat_idx
            sscale_idx = SSCALE_OFFSET + feat_idx

            w(f"        // feature {feat_idx}: dilation={dilation} padding={padding} kernel=({i0},{i1},{i2}) padding1={padding1}")
            w(f"        %fidx_{feat_idx} = arith.constant {feat_idx} : index")
            w(f"        %bidx_{feat_idx} = arith.constant {bias_idx} : index")
            w(f"        %smidx_{feat_idx} = arith.constant {smean_idx} : index")
            w(f"        %ssidx_{feat_idx} = arith.constant {sscale_idx} : index")
            w(f"        %bias_{feat_idx} = memref.load %in_v[%bidx_{feat_idx}] : memref<{COMBINED_IN_LEN}xi32>")
            w(f"        %smean_{feat_idx} = memref.load %in_v[%smidx_{feat_idx}] : memref<{COMBINED_IN_LEN}xi32>")
            w(f"        %sscale_{feat_idx} = memref.load %in_v[%ssidx_{feat_idx}] : memref<{COMBINED_IN_LEN}xi32>")
            w(f"        %dilation_{feat_idx} = arith.constant {dilation} : i32")
            w(f"        %padding_{feat_idx} = arith.constant {padding} : i32")
            w(f"        %ki0_{feat_idx} = arith.constant {i0} : i32")
            w(f"        %ki1_{feat_idx} = arith.constant {i1} : i32")
            w(f"        %ki2_{feat_idx} = arith.constant {i2} : i32")
            w(f"        %pad1_{feat_idx} = arith.constant {padding1} : i32")
            w(f"        %outidx_{feat_idx} = arith.constant {feat_idx} : i32")
            w(
                f"        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_{feat_idx}, "
                f"%smean_{feat_idx}, %sscale_{feat_idx}, %dilation_{feat_idx}, %padding_{feat_idx}, "
                f"%ki0_{feat_idx}, %ki1_{feat_idx}, %ki2_{feat_idx}, %pad1_{feat_idx}, %out_v, %outidx_{feat_idx}) : "
                f"(memref<{COMBINED_IN_LEN}xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, "
                f"memref<{ACTIVE_FEATURES_PER_TILE}xi32>, i32) -> ()"
            )
            w("")

        w(f"        aie.objectfifo.release @in_mem_{col}(Consume, 1)")
        w(f"        aie.objectfifo.release @out_mem_{col}(Produce, 1)")
        w("      }")
        w("      aie.end")
        w('    } {stack_size = 4096 : i32, link_with = "minirocket_transform_kernel.o"}')
    w("")

    # --- Runtime sequence ---
    # Host is responsible for building each tile's combined input buffer
    # as [series | bias_slice | smean_slice | sscale_slice] (COMBINED_IN_LEN
    # elements, matching SERIES_OFFSET/BIAS_OFFSET/SMEAN_OFFSET/SSCALE_OFFSET
    # above) BEFORE calling this kernel -- this is a real, new packing
    # requirement for the host driver, not something the existing project
    # drivers already do; matches the classifier's own precedent of the
    # host doing all data layout work before dispatch (see main_kernels.json
    # convention in dotproduct12_v3.mlir's runtime_sequence).
    w("    // --- HOST RUNTIME SEQUENCE ---")
    w(
        f"    aie.runtime_sequence(%arg0: memref<{COMBINED_IN_LEN * 2}xi32>, "
        f"%arg1: memref<{ACTIVE_FEATURES_PER_TILE * 2}xi32>) {{"
    )
    dma_id = 0
    for col in range(2):
        in_off = col * COMBINED_IN_LEN
        feat_off = col * ACTIVE_FEATURES_PER_TILE
        w(f"      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, {in_off}][1, 1, 1, {COMBINED_IN_LEN}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @in_{col}}} : memref<{COMBINED_IN_LEN * 2}xi32>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, {feat_off}][1, 1, 1, {ACTIVE_FEATURES_PER_TILE}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @out_{col}}} : memref<{ACTIVE_FEATURES_PER_TILE * 2}xi32>")
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
    if TEST_FEATURE_LIMIT is not None:
        print(f"*** TEST-SCALE BUILD *** {ACTIVE_FEATURES_PER_TILE} features/tile "
              f"(real design: {FEATURES_PER_TILE} features/tile, {TOTAL_FEATURES} total) --")
        print(f"    set TEST_FEATURE_LIMIT = None for the full-scale build.")
    else:
        print(f"2 columns x 1 compute tile = 2 compute tiles, {FEATURES_PER_TILE} features/tile, "
              f"{TOTAL_FEATURES} total features")
    print("Path A: one minirocket_feature_transform() call per feature, no cross-feature caching")