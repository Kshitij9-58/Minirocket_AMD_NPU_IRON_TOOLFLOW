#!/usr/bin/env python3
"""
Build the flat 840-feature schedule mapping each output feature index to
its (dilation_index, dilation, kernel_index, padding, padding1) tuple,
matching minirocket.py's transform() iteration order exactly.

Run standalone (no hardware needed) to sanity-check before generating
any MLIR. This is Step 1b/1c of the fusion plan.
"""

import json
from itertools import combinations

DILATIONS = [1, 2, 3, 4, 6, 9, 14, 21, 31]
NUM_FEATURES_PER_DILATION = [2, 1, 1, 1, 1, 1, 1, 1, 1]
SEQ_LENGTH = 251
KERNEL_TAPS = 9


def build_feature_schedule(dilations, num_features_per_dilation, num_kernel_indices=84):
    """
    Matches minirocket.py transform()'s iteration order exactly:
      for dilation_index in range(num_dilations):
          _padding0 = dilation_index % 2
          for kernel_index in range(num_kernels):
              _padding1 = (_padding0 + kernel_index) % 2
              for feature_count in range(num_features_this_dilation):
                  features[..., feature_index_start + feature_count] = PPV(...)
    """
    schedule = []
    for dilation_index, dilation in enumerate(dilations):
        padding0 = dilation_index % 2
        n_feat_this_dilation = num_features_per_dilation[dilation_index]
        padding = ((KERNEL_TAPS - 1) * dilation) // 2
        for kernel_index in range(num_kernel_indices):
            padding1 = (padding0 + kernel_index) % 2
            for _ in range(n_feat_this_dilation):
                schedule.append({
                    "dilation_index": dilation_index,
                    "dilation": dilation,
                    "kernel_index": kernel_index,
                    "padding": padding,
                    "padding1": padding1,
                })
    return schedule


def main():
    schedule = build_feature_schedule(DILATIONS, NUM_FEATURES_PER_DILATION)

    print(f"Total schedule entries: {len(schedule)}")
    assert len(schedule) == 840, f"Expected 840, got {len(schedule)}"

    # Cross-check: count of entries per dilation_index should match
    # NUM_FEATURES_PER_DILATION * 84 (num_kernel_indices)
    for di, dilation in enumerate(DILATIONS):
        count = sum(1 for e in schedule if e["dilation_index"] == di)
        expected = NUM_FEATURES_PER_DILATION[di] * 84
        status = "OK" if count == expected else "MISMATCH"
        print(f"dilation_index={di} dilation={dilation:3d} count={count:4d} expected={expected:4d} [{status}]")
        assert count == expected

    # Cross-check: padding values must satisfy end = SEQ_LENGTH - padding > 0
    for e in schedule:
        end = SEQ_LENGTH - e["padding"]
        assert end > 0, f"Invalid padding {e['padding']} for dilation {e['dilation']} (SEQ_LENGTH={SEQ_LENGTH})"
    print(f"\nAll padding values valid (end > 0 for SEQ_LENGTH={SEQ_LENGTH})")

    # padding1 alternation sanity check: for dilation_index=0 (padding0=0),
    # padding1 should alternate 0,1,0,1,... across kernel_index 0..83
    di0_entries = [e for e in schedule if e["dilation_index"] == 0]
    # dilation_index=0 has 2 features per kernel_index, so kernel_index
    # repeats twice consecutively in this slice
    print("\nFirst 10 entries for dilation_index=0 (padding0=0, 2 features/kernel_index):")
    for e in di0_entries[:10]:
        print(f"  kernel_index={e['kernel_index']:3d} padding1={e['padding1']}")

    # Split into 2 tiles, 420 features each (matches existing proven
    # transform's FEATURES_PER_CORE split)
    FEATURES_PER_CORE = 420
    tile0_schedule = schedule[:FEATURES_PER_CORE]
    tile1_schedule = schedule[FEATURES_PER_CORE:]
    assert len(tile0_schedule) == 420
    assert len(tile1_schedule) == 420
    print(f"\nTile split: tile0={len(tile0_schedule)} features (dilation_index "
          f"{tile0_schedule[0]['dilation_index']}..{tile0_schedule[-1]['dilation_index']}), "
          f"tile1={len(tile1_schedule)} features (dilation_index "
          f"{tile1_schedule[0]['dilation_index']}..{tile1_schedule[-1]['dilation_index']})")

    # How many DISTINCT (dilation_index, kernel_index) pairs does each
    # tile need to compute C_alpha/C_gamma for? This is the real cost
    # driver flagged in Step 1e (Path A vs Path B).
    tile0_distinct = len(set((e["dilation_index"], e["kernel_index"]) for e in tile0_schedule))
    tile1_distinct = len(set((e["dilation_index"], e["kernel_index"]) for e in tile1_schedule))
    print(f"\nDistinct (dilation_index, kernel_index) pairs needing C_alpha/C_gamma recompute:")
    print(f"  tile0: {tile0_distinct} distinct pairs for {len(tile0_schedule)} features")
    print(f"  tile1: {tile1_distinct} distinct pairs for {len(tile1_schedule)} features")
    print(f"  (Path A recomputes once per FEATURE -- {len(tile0_schedule)}/{len(tile1_schedule)} times;")
    print(f"   Path B recomputes once per DISTINCT PAIR -- {tile0_distinct}/{tile1_distinct} times)")

    # Save schedule to JSON for reuse when generating the actual MLIR kernel
    with open("feature_schedule.json", "w") as f:
        json.dump(schedule, f, indent=2)
    print("\nSaved full schedule to feature_schedule.json")


if __name__ == "__main__":
    main()
