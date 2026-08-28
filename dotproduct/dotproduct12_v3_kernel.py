#!/usr/bin/env python3
"""
dotproduct12_v3_kernel.py

STATUS: DRAFT, UNBUILT, UNVERIFIED. Not yet run through aiecc.py, not
compiled, not run on hardware.

Generates MLIR for the 12-tile dot-product classifier, built as a
DIRECT STRUCTURAL MIRROR of the real, working GEMM-12 MLIR (the actual
source behind final-12.xclbin, confirmed against the real file -- device
`npu1`, 3 columns x 4 compute tiles, per-column objectfifo.link gather
pattern, per-column single DMA dispatch). Only the per-tile compute and
buffer shapes change: 70-element dot-product chunks instead of padded
64x64/64x32 GEMM blocks.

WHAT IS THE SAME AS THE REAL GEMM FILE (verified structural mirror):
  - Device symbol: npu1
  - Tile layout: 3 columns (0,1,2), each with shim(row0)/mem(row1)/
    4 compute tiles(rows 2-5)
  - Per-column pattern: shim -> mem -> broadcast to 4 compute tiles
    (inA/inB -> memA/memB, link'd), each compute tile produces into its
    own memC_{col}_{row}, all 4 link-merged into one outC_col{N} per
    column, DMA'd out once per column
  - Core loop shape: outer infinite loop, inner "repeat count" loop,
    zero-accumulator call then per-iteration compute call inside an
    innermost loop, objectfifo acquire/release around each step
  - Runtime sequence shape: one DMA-in per column per input tensor, one
    DMA-out per column, all issued together, all waited on together

WHAT IS DIFFERENT (the actual experimental variable):
  - Per-tile payload: CHUNK_SIZE=70 elements (real slice of the 840-wide
    MiniROCKET feature vector) instead of a 64x64=4096-element padded
    block (~98% zero, per the project's own documented Limitations).
  - Compute function: dotproduct_partial_bf16_f32 (a 70-element
    multiply-accumulate) instead of matmul_bf16_f32.
  - Output per tile: a single f32 partial sum instead of a 64x32xf32
    block. Each column's 4 tiles' partials are link-merged (gathered,
    NOT summed on-chip -- mirrors GEMM's gather-only merge pattern
    exactly) into a 4-element outC_col{N} buffer. Final summation of
    all 12 partials (4 per column x 3 columns) happens on the HOST,
    matching how GEMM's host script already does
    `total_class_score += float(raw_str)` per chunk today.

UNVERIFIED / HIGHEST-RISK ITEMS IN THIS FILE (verify before building):
  1. The real GEMM core loop's exact repeat-count semantics (128 outer,
     8 inner = 1024 matmul calls per tile per dispatch) reflect GEMM's
     own batching scheme for its specific workload size. This generator
     uses a SINGLE compute call per tile per dispatch (each tile only
     has one 70-element chunk to process, no repeat/accumulate-over-many-
     chunks structure needed) -- this is a deliberate simplification,
     not a copy-paste of GEMM's loop counts, since the underlying access
     pattern is different (no equivalent of GEMM's need to iterate 8x
     over sub-blocks). If dispatch behavior looks wrong once built,
     this loop-count simplification is the first place to check.
  2. dimensionsToStream access-pattern annotations on GEMM's memA/memB/
     outC objectfifos (the `<size=.., stride=..>` lists) encode specific
     strided-access patterns tuned for 64x64/64x32 block shapes. This
     generator OMITS those annotations for the 70-element dot-product
     buffers (uses plain objectfifo declarations without
     dimensionsToStream), since a flat 70-element chunk doesn't need
     GEMM's multi-dimensional block-streaming reshape. UNVERIFIED that
     omitting this entirely (rather than providing a trivial/identity
     version) is correct MLIR-AIE -- check against mlir-aie docs/examples
     for objectfifo declarations without dimensionsToStream if the build
     fails here specifically.
  3. Runtime sequence buffer sizing: real GEMM's args are
     memref<3145728xbf16/f32> (3 columns x 4 tiles x 64x64 or 64x32,
     the FULL padded multi-column buffer). This generator computes the
     equivalent total size for 3 columns x 4 tiles x 70 elements
     (constants below) -- arithmetic double-checked in this script but
     NOT verified against a real compile.
"""

NUM_COLS = 3
TILES_PER_COL = 4
CHUNK_SIZE = 70               # real 840/12 split, no padding
FEATURE_COUNT_TOTAL = 840
DEVICE_SYMBOL = "npu1"        # CONFIRMED from real final-12.xclbin MLIR

# Per-column, per-tensor buffer size (features OR weights), matching
# GEMM's per-column slicing convention (each column handles 4 tiles'
# worth of data in one contiguous region of the flat host buffer).
PER_COL_ELEMENTS = TILES_PER_COL * CHUNK_SIZE          # 4 * 70 = 280
TOTAL_ELEMENTS = NUM_COLS * PER_COL_ELEMENTS            # 3 * 280 = 840 (checks out exactly)
PER_COL_OUT_ELEMENTS = TILES_PER_COL                    # 4 partials out per column
TOTAL_OUT_ELEMENTS = NUM_COLS * PER_COL_OUT_ELEMENTS    # 12 partials total


def generate_mlir():
    lines = []
    w = lines.append

    w("module {")
    w(f"  aie.device({DEVICE_SYMBOL}) {{")
    w(f"    func.func private @zero_f32_scalar(memref<1xf32>)")
    w(f"    func.func private @dotproduct_partial_bf16_f32(memref<{CHUNK_SIZE}xbf16>, memref<{CHUNK_SIZE}xbf16>, memref<1xf32>)")
    w("")

    # --- Tiles: identical coordinates to the real GEMM file ---
    for col in range(NUM_COLS):
        w(f"    %shim_noc_tile_{col}_0 = aie.tile({col}, 0)")
        w(f"    %mem_tile_{col}_1 = aie.tile({col}, 1)")
        for r in range(TILES_PER_COL):
            row = 2 + r
            w(f"    %tile_{col}_{row} = aie.tile({col}, {row})")
    w("")

    # --- Per-column objectfifos: mirrors inA/memA, inB/memB, memC_*,
    # outC_col{N} pattern exactly, shapes swapped to CHUNK_SIZE ---
    for col in range(NUM_COLS):
        w(f"    // --- ROUTING: COLUMN {col} ---")
        w(f"    aie.objectfifo @inF_col{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{PER_COL_ELEMENTS}xbf16>>")
        w(f"    aie.objectfifo @memF_col{col}(%mem_tile_{col}_1, {{" +
          ", ".join(f"%tile_{col}_{2+r}" for r in range(TILES_PER_COL)) +
          f"}}, 2 : i32) : !aie.objectfifo<memref<{CHUNK_SIZE}xbf16>>")
        w(f"    aie.objectfifo.link [@inF_col{col}] -> [@memF_col{col}]([] [])")

        w(f"    aie.objectfifo @inW_col{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{PER_COL_ELEMENTS}xbf16>>")
        w(f"    aie.objectfifo @memW_col{col}(%mem_tile_{col}_1, {{" +
          ", ".join(f"%tile_{col}_{2+r}" for r in range(TILES_PER_COL)) +
          f"}}, 2 : i32) : !aie.objectfifo<memref<{CHUNK_SIZE}xbf16>>")
        w(f"    aie.objectfifo.link [@inW_col{col}] -> [@memW_col{col}]([] [])")

        for r in range(TILES_PER_COL):
            row = 2 + r
            w(f"    aie.objectfifo @memC_{col}_{row}(%tile_{col}_{row}, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<1xf32>>")

        w(f"    aie.objectfifo @outC_col{col}(%mem_tile_{col}_1, {{%shim_noc_tile_{col}_0}}, 2 : i32) : !aie.objectfifo<memref<{PER_COL_OUT_ELEMENTS}xf32>>")
        merge_srcs = ", ".join(f"@memC_{col}_{2+r}" for r in range(TILES_PER_COL))
        zeros = ", ".join("0" for _ in range(TILES_PER_COL))
        w(f"    aie.objectfifo.link [{merge_srcs}] -> [@outC_col{col}]([{zeros}] [])")
        w("")

    # --- Cores: one per compute tile, mirrors GEMM's zero-then-compute
    # loop shape, simplified to a single compute call per dispatch (see
    # risk item #1 in the module docstring above) ---
    w("    // --- CORES ---")
    for col in range(NUM_COLS):
        for r in range(TILES_PER_COL):
            row = 2 + r
            w(f"    %core_{col}_{row} = aie.core(%tile_{col}_{row}) {{")
            w("      %c0 = arith.constant 0 : index")
            w("      %c4294967295 = arith.constant 4294967295 : index")
            w("      %c1 = arith.constant 1 : index")
            w("      scf.for %arg0 = %c0 to %c4294967295 step %c1 {")
            w(f"        %0 = aie.objectfifo.acquire @memC_{col}_{row}(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>")
            w("        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>")
            w("        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()")
            w(f"        %2 = aie.objectfifo.acquire @memF_col{col}(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK_SIZE}xbf16>>")
            w(f"        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<{CHUNK_SIZE}xbf16>> -> memref<{CHUNK_SIZE}xbf16>")
            w(f"        %4 = aie.objectfifo.acquire @memW_col{col}(Consume, 1) : !aie.objectfifosubview<memref<{CHUNK_SIZE}xbf16>>")
            w(f"        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<{CHUNK_SIZE}xbf16>> -> memref<{CHUNK_SIZE}xbf16>")
            w(f"        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<{CHUNK_SIZE}xbf16>, memref<{CHUNK_SIZE}xbf16>, memref<1xf32>) -> ()")
            w(f"        aie.objectfifo.release @memF_col{col}(Consume, 1)")
            w(f"        aie.objectfifo.release @memW_col{col}(Consume, 1)")
            w(f"        aie.objectfifo.release @memC_{col}_{row}(Produce, 1)")
            w("      }")
            w("      aie.end")
            w("    } {stack_size = 2048 : i32}")
    w("")

    # --- Runtime sequence: one DMA-in per column per tensor, one
    # DMA-out per column, all issued together then all waited on,
    # mirroring GEMM's flat-buffer-with-offset convention exactly ---
    w("    // --- HOST RUNTIME SEQUENCE ---")
    w(f"    aie.runtime_sequence(%arg0: memref<{TOTAL_ELEMENTS}xbf16>, %arg1: memref<{TOTAL_ELEMENTS}xbf16>, %arg2: memref<{TOTAL_OUT_ELEMENTS}xf32>) {{")
    dma_id = 0
    for col in range(NUM_COLS):
        f_off = col * PER_COL_ELEMENTS
        out_off = col * PER_COL_OUT_ELEMENTS
        w(f"      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, {f_off}][1, 1, 1, {PER_COL_ELEMENTS}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @inF_col{col}}} : memref<{TOTAL_ELEMENTS}xbf16>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, {f_off}][1, 1, 1, {PER_COL_ELEMENTS}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @inW_col{col}}} : memref<{TOTAL_ELEMENTS}xbf16>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, {out_off}][1, 1, 1, {PER_COL_OUT_ELEMENTS}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @outC_col{col}}} : memref<{TOTAL_OUT_ELEMENTS}xf32>")
        dma_id += 1
    for col in range(NUM_COLS):
        w(f"      aiex.npu.dma_wait {{symbol = @outC_col{col}}}")
    w("    }")
    w("  }")
    w("}")

    return "\n".join(lines)


if __name__ == "__main__":
    assert TOTAL_ELEMENTS == FEATURE_COUNT_TOTAL, \
        f"Sanity check failed: {NUM_COLS} cols x {TILES_PER_COL} tiles x {CHUNK_SIZE} = {TOTAL_ELEMENTS}, expected {FEATURE_COUNT_TOTAL}"

    mlir_text = generate_mlir()
    out_path = "dotproduct12_v3.mlir"
    with open(out_path, "w") as f:
        f.write(mlir_text)
    print(f"Wrote {out_path}")
    print(f"{NUM_COLS} columns x {TILES_PER_COL} compute tiles = {NUM_COLS*TILES_PER_COL} compute tiles")
    print(f"Per-column chunk: {PER_COL_ELEMENTS} elements ({TILES_PER_COL} tiles x {CHUNK_SIZE})")
    print(f"Total: {TOTAL_ELEMENTS} elements in, {TOTAL_OUT_ELEMENTS} partials out")
    print("STATUS: DRAFT. Not yet run through aiecc.py. See docstring risk items 1-3 before building.")
