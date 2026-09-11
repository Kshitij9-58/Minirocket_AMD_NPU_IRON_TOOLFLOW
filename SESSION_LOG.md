# on-chip-execution — Session Log

Branch: `on-chip-execution` (off `main`). Repo:
https://github.com/Kshitij9-58/Minirocket_AMD_NPU_IRON_TOOLFLOW

## Goal
Fuse the MiniROCKET transform stage and the 12-tile dot-product
classifier into a single on-chip pipeline (no CPU round-trip), with the
transform stage doing REAL MiniROCKET math (dilation + 84 kernel-index-
triples + padding-alternated PPV + normalization) rather than the
existing contiguous-window stand-in whose output was previously
discarded in favor of CPU-computed features.

## Status: Step 1 (real transform kernel) — FULLY VERIFIED ON HARDWARE AT FULL SCALE ✅
Commits through `d4e6062` on `on-chip-execution`.

**Full-scale (420 features/tile, 840 total) result: 840/840 exact
bit-for-bit matches** between NPU output and CPU reference. Measured
dispatch latency: **14.28 ms/dispatch** (100 iterations, real hardware
measurement, not modeled).

This confirms the on-chip kernel correctly implements the COMPLETE real
MiniROCKET transform: dilated `C_alpha`/`C_gamma` accumulation across
all 9 dilation levels (1,2,3,4,6,9,14,21,31), all 84 kernel-index-triple
combinations, padding-alternated PPV pooling, and standardization +
clipping — the actual algorithm, not an approximation, at the actual
design scale.

### The journey: two architectures

**Architecture 1 (per-feature call sites):** one MLIR-level function
call per feature, 840 distinct call sites at full scale. Worked
correctly at small scale (8 features/tile: 16/16 exact matches, 0.592
ms/dispatch) but failed at full scale with "Overflow of program
memory" — too much unrolled instruction code per tile.

**Architecture 2 (data-driven, current):** ONE function call per tile
(`minirocket_transform_tile`), looping internally in compiled C++ over
a parameter array `[dilation, padding, kernel_i0, kernel_i1, kernel_i2,
padding1]` per feature, delivered as DATA rather than baked into
separate call sites. Uses `aie.buffer` (confirmed via official mlir-aie
documentation as the correct static-allocation replacement for
`memref.alloc`) for local working memory.

Result: generates the SAME ~153-line MLIR regardless of feature count
(8 or 420) — program memory is now fully decoupled from feature count.
Verified correct AND faster at matching scale:
- 8 features/tile: 16/16 exact matches, **0.42 ms/dispatch** (vs 0.59ms
  for architecture 1)
- 420 features/tile (full design): **840/840 exact matches, 14.28
  ms/dispatch**

### Files (all in `dotproduct/`, committed)
- `minirocket_transform_kernel.cc` — data-driven C++. Two functions:
  `compute_one_feature()` (the verified per-feature math, unchanged
  from architecture 1) and `minirocket_transform_tile()` (loops over
  the parameter array, calling `compute_one_feature()` per feature).
- `minirocket_transform_kernel.py` — data-driven MLIR generator.
  `TEST_FEATURE_LIMIT` switch: `8` for validation, `None` for full
  420/tile design. Uses `aie.buffer` at device scope, not
  `memref.alloc()`.
- `minirocket_transform_v1.mlir` — generated MLIR (currently reflects
  whichever scale was last generated — check `TEST_FEATURE_LIMIT` in
  the `.py` and regenerate if needed).
- `prep_and_verify_test8.py` / `prep_and_verify_full420.py` — build the
  combined input buffer (series + bias + scaler_mean + scaler_scale +
  parameter array) and compute the CPU reference in matching
  fixed-point arithmetic, at test/full scale respectively.
- `transform_test8_runner.cpp` / `transform_full420_runner.cpp` — XRT
  dispatch runners, modeled on the proven `dotprod12_runner.cpp`
  pattern, sized for their respective scales.
- `transform_datadriven_test8.xclbin` / `_sequence.bin` — verified
  8-feature/tile build (16/16 matches, 0.42ms).
- `transform_full420.xclbin` / `_sequence.bin` — verified 420-
  feature/tile FULL DESIGN build (840/840 matches, 14.28ms). **This is
  the artifact to use going forward.**

### Key bugs found and fixed (chronological, all confirmed via real
### hardware errors, not guessed)
1. `memref.subview`-typed call arguments → MLIR type mismatch. Fixed:
   pass buffers directly, no subview.
2. Shim tile DMA output channel limit exceeded (4 input ObjectFifos +
   1 output = 5 streams). Fixed: consolidated into 1 combined input
   buffer per tile.
3. Fully-unrolled 251-iteration series-copy loop bloated code size.
   Fixed: real `scf.for` runtime loop.
4. `memref.alloc()` requires a libc heap pool (`_ZL4pool`) with zero
   free space in the tile's 4x16KB memory banks (confirmed via
   comparison against the proven classifier's map file, which has no
   `_ZL4pool` references since it never calls `memref.alloc()`). Fixed
   in architecture 1 by removing scratch-copy allocations; fully
   eliminated in architecture 2 via `aie.buffer`.
5. 9x251 `C_gamma` table wastefully large (~9KB/call, only 3 of 9 rows
   ever read). Fixed: direct running-sum accumulation (~2.94KB/call).
6. `link_with` attribute silently ignored when placed on `func.func`
   declarations in this toolchain version; must go on `aie.core`'s
   attribute dict. Confirmed via the literal linker command showing an
   empty gap where the `.o` path should have been.
7. Normalization scaling bug: `diff / denom` on two already-SCALE-
   scaled values cancels the scale factor. Fixed: `(diff * SCALE) /
   denom`.
8. **"Overflow of program memory" at full scale** (840 unrolled call
   sites, architecture 1) — fixed via the data-driven redesign
   (architecture 2): one call site per tile, parameters as data.

### Build reliability
Architecture 1's full `aiecc.py` compile was non-deterministic —
succeeded ~30-40% of identical-input attempts. Architecture 2's builds
(both scales) have succeeded cleanly and quickly (under a minute each)
on every attempt since the redesign. Not independently confirmed across
many repeated attempts, but a strong improvement observed so far.

## Next steps
1. **Fuse the verified 2-tile transform with the proven 12-tile
   dot-product classifier** into a single 14-tile on-chip pipeline
   (Step 4/5 of `FUSION_PLAN.md`): bridge transform output directly to
   classifier input via `object_fifo_link` through a mem_tile (420-wide
   transform output needs reshaping to the classifier's 70-wide
   per-tile inputs), combine into one `aie.device()` block, one
   dispatch, no CPU round-trip.
2. Measure the fused pipeline's real per-sample latency and compare
   against today's two-dispatch baseline (14.28ms transform +
   classifier dispatch) — likely dominated by eliminated subprocess/
   dispatch overhead rather than compute time.
3. Update the IEEEtran paper draft with real transform-stage numbers
   (840/840 correctness, 14.28ms measured) now that they exist.
