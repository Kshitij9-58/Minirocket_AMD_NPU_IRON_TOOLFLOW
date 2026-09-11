#!/usr/bin/env python3
"""
minirocket_transform_kernel.py

DATA-DRIVEN VERSION -- replaces the per-feature-call-site design that
hit "Overflow of program memory" at full scale (420 features/tile, 840
unrolled call sites). This version makes ONE function call per tile
(minirocket_transform_tile), which loops internally in compiled C++ over
a parameter array delivered as DATA, not as 420 separate MLIR-level call
sites. Program memory usage is now roughly constant regardless of
FEATURES_PER_TILE.

The per-feature MATH is unchanged and already verified correct against
real NPU hardware output (16/16 exact bit-for-bit matches at test scale,
see dotproduct/transform_test8.xclbin and SESSION_LOG.md) -- only the
CALLING STRUCTURE changed.

Style: raw string-templated MLIR (matches dotproduct12_v3_kernel.py's
proven pattern), NOT the aie.dialects.aie/mlir_mod_ctx Python DSL.
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
FEATURES_PER_TILE = 420
DEVICE_SYMBOL = "npu1"
PARAMS_PER_FEATURE = 6  # dilation, padding, kernel_i0, kernel_i1, kernel_i2, padding1

KERNEL_INDICES = list(combinations(range(KERNEL_TAPS), 3))
assert len(KERNEL_INDICES) == NUM_KERNEL_INDICES

# TEST_FEATURE_LIMIT: if set, each tile only processes this many
# features (for validation builds). None = full 420/tile design.
TEST_FEATURE_LIMIT = 8
ACTIVE_FEATURES_PER_TILE = TEST_FEATURE_LIMIT if TEST_FEATURE_LIMIT is not None else FEATURES_PER_TILE

# Absolute path to the compiled .o file, resolved at GENERATION time.
# MUST run this script from the same directory where
# minirocket_transform_kernel.o will be compiled (i.e. on wolverine,
# from inside transform_v1_build/) -- confirmed necessary tonight since
# link_with resolution needs either an absolute path or to match
# xchesscc's actual invocation directory, and a bare filename failed
# non-deterministically across multiple real compile attempts.
LINK_WITH_OBJ_PATH = os.path.join(os.getcwd(), "minirocket_transform_kernel.o")


def build_feature_schedule():
    """Matches minirocket.py transform()'s iteration order exactly."""
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
                    "dilation": dilation, "padding": padding,
                    "kernel_i0": i0, "kernel_i1": i1, "kernel_i2": i2,
                    "padding1": padding1,
                })
    assert len(schedule) == TOTAL_FEATURES, f"Expected {TOTAL_FEATURES}, got {len(schedule)}"
    return schedule


def generate_mlir():
    schedule = build_feature_schedule()
    tile_schedules = [
        schedule[:FEATURES_PER_TILE][:ACTIVE_FEATURES_PER_TILE],
        schedule[FEATURES_PER_TILE:][:ACTIVE_FEATURES_PER_TILE],
    ]

    # Combined input layout per tile:
    # [series(SEQ_LENGTH) | bias(F) | smean(F) | sscale(F) | params(F*6)]
    F = ACTIVE_FEATURES_PER_TILE
    SERIES_OFFSET = 0
    BIAS_OFFSET = SEQ_LENGTH
    SMEAN_OFFSET = SEQ_LENGTH + F
    SSCALE_OFFSET = SEQ_LENGTH + 2 * F
    PARAMS_OFFSET = SEQ_LENGTH + 3 * F
    COMBINED_IN_LEN = SEQ_LENGTH + 3 * F + F * PARAMS_PER_FEATURE

    lines = []
    w = lines.append

    w("module {")
    w(f"  aie.device({DEVICE_SYMBOL}) {{")
    w(f"    func.func private @zero_i32_scalar(memref<1xi32>)")
    w(
        f'    func.func private @minirocket_transform_tile('
        f"memref<{SEQ_LENGTH}xi32>, memref<{F}xi32>, memref<{F}xi32>, "
        f"memref<{F}xi32>, memref<{F * PARAMS_PER_FEATURE}xi32>, i32, "
        f"memref<{F}xi32>)"
    )
    w("")

    for col in range(2):
        w(f"    %shim_noc_tile_{col}_0 = aie.tile({col}, 0)")
        w(f"    %mem_tile_{col}_1 = aie.tile({col}, 1)")
        w(f"    %tile_{col}_2 = aie.tile({col}, 2)")
    w("")

    for col in range(2):
        w(f"    %series_local_{col} = aie.buffer(%tile_{col}_2) {{sym_name = \"series_local_{col}\"}} : memref<{SEQ_LENGTH}xi32>")
        w(f"    %bias_local_{col} = aie.buffer(%tile_{col}_2) {{sym_name = \"bias_local_{col}\"}} : memref<{F}xi32>")
        w(f"    %smean_local_{col} = aie.buffer(%tile_{col}_2) {{sym_name = \"smean_local_{col}\"}} : memref<{F}xi32>")
        w(f"    %sscale_local_{col} = aie.buffer(%tile_{col}_2) {{sym_name = \"sscale_local_{col}\"}} : memref<{F}xi32>")
        w(f"    %params_local_{col} = aie.buffer(%tile_{col}_2) {{sym_name = \"params_local_{col}\"}} : memref<{F * PARAMS_PER_FEATURE}xi32>")
    w("")

    for col in range(2):
        w(f"    // --- ROUTING: COLUMN {col} (transform tile {col}) ---")
        w(f"    aie.objectfifo @in_{col}(%shim_noc_tile_{col}_0, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{COMBINED_IN_LEN}xi32>>")
        w(f"    aie.objectfifo @in_mem_{col}(%mem_tile_{col}_1, {{%tile_{col}_2}}, 2 : i32) : !aie.objectfifo<memref<{COMBINED_IN_LEN}xi32>>")
        w(f"    aie.objectfifo.link [@in_{col}] -> [@in_mem_{col}]([] [])")

        w(f"    aie.objectfifo @out_mem_{col}(%tile_{col}_2, {{%mem_tile_{col}_1}}, 2 : i32) : !aie.objectfifo<memref<{F}xi32>>")
        w(f"    aie.objectfifo @out_{col}(%mem_tile_{col}_1, {{%shim_noc_tile_{col}_0}}, 2 : i32) : !aie.objectfifo<memref<{F}xi32>>")
        w(f"    aie.objectfifo.link [@out_mem_{col}] -> [@out_{col}]([] [])")
        w("")

    w("    // --- CORES ---")
    for col in range(2):
        w(f"    %core_{col}_2 = aie.core(%tile_{col}_2) {{")
        w("      %c0 = arith.constant 0 : index")
        w("      %c4294967295 = arith.constant 4294967295 : index")
        w("      %c1 = arith.constant 1 : index")
        w(f"      %c_series_off = arith.constant {SERIES_OFFSET} : index")
        w(f"      %c_bias_off = arith.constant {BIAS_OFFSET} : index")
        w(f"      %c_smean_off = arith.constant {SMEAN_OFFSET} : index")
        w(f"      %c_sscale_off = arith.constant {SSCALE_OFFSET} : index")
        w(f"      %c_params_off = arith.constant {PARAMS_OFFSET} : index")
        w(f"      %c_seq_len = arith.constant {SEQ_LENGTH} : index")
        w(f"      %c_num_feat = arith.constant {F} : i32")
        w("")

        w("      scf.for %arg0 = %c0 to %c4294967295 step %c1 {")
        w(f"        %in_elem = aie.objectfifo.acquire @in_mem_{col}(Consume, 1) : !aie.objectfifosubview<memref<{COMBINED_IN_LEN}xi32>>")
        w(f"        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<{COMBINED_IN_LEN}xi32>> -> memref<{COMBINED_IN_LEN}xi32>")
        w(f"        %out = aie.objectfifo.acquire @out_mem_{col}(Produce, 1) : !aie.objectfifosubview<memref<{F}xi32>>")
        w(f"        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<{F}xi32>> -> memref<{F}xi32>")
        w("")
        w(f"        scf.for %s_i = %c0 to %c_seq_len step %c1 {{")
        w(f"          %s_src_idx = arith.addi %c_series_off, %s_i : index")
        w(f"          %s_val = memref.load %in_v[%s_src_idx] : memref<{COMBINED_IN_LEN}xi32>")
        w(f"          memref.store %s_val, %series_local_{col}[%s_i] : memref<{SEQ_LENGTH}xi32>")
        w(f"        }}")
        w("")

        w(f"        %c_F = arith.constant {F} : index")
        w(f"        scf.for %b_i = %c0 to %c_F step %c1 {{")
        w(f"          %b_src_idx = arith.addi %c_bias_off, %b_i : index")
        w(f"          %b_val = memref.load %in_v[%b_src_idx] : memref<{COMBINED_IN_LEN}xi32>")
        w(f"          memref.store %b_val, %bias_local_{col}[%b_i] : memref<{F}xi32>")
        w(f"          %sm_src_idx = arith.addi %c_smean_off, %b_i : index")
        w(f"          %sm_val = memref.load %in_v[%sm_src_idx] : memref<{COMBINED_IN_LEN}xi32>")
        w(f"          memref.store %sm_val, %smean_local_{col}[%b_i] : memref<{F}xi32>")
        w(f"          %ss_src_idx = arith.addi %c_sscale_off, %b_i : index")
        w(f"          %ss_val = memref.load %in_v[%ss_src_idx] : memref<{COMBINED_IN_LEN}xi32>")
        w(f"          memref.store %ss_val, %sscale_local_{col}[%b_i] : memref<{F}xi32>")
        w(f"        }}")
        w("")

        w(f"        %c_Pn = arith.constant {F * PARAMS_PER_FEATURE} : index")
        w(f"        scf.for %p_i = %c0 to %c_Pn step %c1 {{")
        w(f"          %p_src_idx = arith.addi %c_params_off, %p_i : index")
        w(f"          %p_val = memref.load %in_v[%p_src_idx] : memref<{COMBINED_IN_LEN}xi32>")
        w(f"          memref.store %p_val, %params_local_{col}[%p_i] : memref<{F * PARAMS_PER_FEATURE}xi32>")
        w(f"        }}")
        w("")

        w(
            f"        func.call @minirocket_transform_tile(%series_local_{col}, %bias_local_{col}, "
            f"%smean_local_{col}, %sscale_local_{col}, %params_local_{col}, %c_num_feat, %out_v) : "
            f"(memref<{SEQ_LENGTH}xi32>, memref<{F}xi32>, memref<{F}xi32>, "
            f"memref<{F}xi32>, memref<{F * PARAMS_PER_FEATURE}xi32>, i32, "
            f"memref<{F}xi32>) -> ()"
        )
        w("")

        w(f"        aie.objectfifo.release @in_mem_{col}(Consume, 1)")
        w(f"        aie.objectfifo.release @out_mem_{col}(Produce, 1)")
        w("      }")
        w("      aie.end")
        w('    } {stack_size = 4096 : i32, link_with = "minirocket_transform_kernel.o"}')
    w("")

    w("    // --- HOST RUNTIME SEQUENCE ---")
    w(
        f"    aie.runtime_sequence(%arg0: memref<{COMBINED_IN_LEN * 2}xi32>, "
        f"%arg1: memref<{F * 2}xi32>) {{"
    )
    dma_id = 0
    for col in range(2):
        in_off = col * COMBINED_IN_LEN
        feat_off = col * F
        w(f"      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, {in_off}][1, 1, 1, {COMBINED_IN_LEN}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @in_{col}}} : memref<{COMBINED_IN_LEN * 2}xi32>")
        dma_id += 1
        w(f"      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, {feat_off}][1, 1, 1, {F}][0, 0, 0, 1]) {{id = {dma_id} : i64, metadata = @out_{col}}} : memref<{F * 2}xi32>")
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
    print(f"DATA-DRIVEN design: 1 function call site per tile (not {ACTIVE_FEATURES_PER_TILE} per tile as before)")
    print(f"{ACTIVE_FEATURES_PER_TILE} features/tile, "
          f"{2 * ACTIVE_FEATURES_PER_TILE} total features processed per dispatch")