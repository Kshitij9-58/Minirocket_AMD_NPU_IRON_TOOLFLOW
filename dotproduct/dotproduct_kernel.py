#!/usr/bin/env python3
# dotproduct_kernel.py
# Minimal single-tile MLIR generator for the classifier redesign.
#
# STATUS: untested starting point, not yet compiled or run. Structurally
# modeled on minirocket_kernel.py's GEMM generator (object_fifo layout,
# runtime_sequence pattern), but the exact npu_dma_memcpy_nd arguments and
# object_fifo sizing here are a first pass, adjust as needed once you run
# this through aiecc.py and see what actually compiles.
#
# Scope: ONE tile, ONE sample, ONE class's dot product. No batching, no
# spatial split. Get this compiling and matching the CPU float score before
# extending to multiple classes or multiple tiles.

import numpy as np
from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.iron.controlflow import range_
from aie.iron.dtype import str_to_dtype

FEATURE_COUNT = 840  # matches TOTAL_FEATURES in the deployment script


def dotproduct_kernel(dev="npu"):
    dtype_in = str_to_dtype("bf16")
    dtype_out = str_to_dtype("f32")

    dev_ty = AIEDevice.npu1_1col if dev == "npu" else AIEDevice.npu2

    with mlir_mod_ctx() as ctx:

        @device(dev_ty)
        def device_body():
            vec_ty = np.ndarray[(FEATURE_COUNT,), np.dtype[dtype_in]]
            score_ty = np.ndarray[(1,), np.dtype[dtype_out]]

            # External C functions from dotproduct_bf16_f32.cc
            dotproduct = external_func(
                "dotproduct_bf16_f32",
                inputs=[vec_ty, vec_ty, score_ty],
            )
            zero_score = external_func("zero_score", inputs=[score_ty])

            shim_tile = tile(0, 0)
            mem_tile = tile(0, 1)
            compute_tile = tile(0, 2)

            # Feature vector: host -> mem tile -> compute tile
            inF = object_fifo("inF", shim_tile, mem_tile, 2, vec_ty)
            memF = object_fifo("memF", mem_tile, compute_tile, 2, vec_ty)
            object_fifo_link(inF, memF)

            # Weight vector: host -> mem tile -> compute tile
            inW = object_fifo("inW", shim_tile, mem_tile, 2, vec_ty)
            memW = object_fifo("memW", mem_tile, compute_tile, 2, vec_ty)
            object_fifo_link(inW, memW)

            # Score: compute tile -> mem tile -> host
            outS = object_fifo("outS", compute_tile, mem_tile, 2, score_ty)
            finalS = object_fifo("finalS", mem_tile, shim_tile, 2, score_ty)
            object_fifo_link(outS, finalS)

            @core(compute_tile, stack_size=0x800)
            def core_body():
                for _ in range_(0xFFFFFFFF):
                    elem_f = memF.acquire(ObjectFifoPort.Consume, 1)
                    elem_w = memW.acquire(ObjectFifoPort.Consume, 1)
                    elem_out = outS.acquire(ObjectFifoPort.Produce, 1)

                    zero_score(elem_out)
                    dotproduct(elem_f, elem_w, elem_out)

                    memF.release(ObjectFifoPort.Consume, 1)
                    memW.release(ObjectFifoPort.Consume, 1)
                    outS.release(ObjectFifoPort.Produce, 1)

            @runtime_sequence(
                np.ndarray[(FEATURE_COUNT,), np.dtype[dtype_in]],
                np.ndarray[(FEATURE_COUNT,), np.dtype[dtype_in]],
                np.ndarray[(1,), np.dtype[dtype_out]],
            )
            def sequence(F, W, S):
                npu_dma_memcpy_nd(metadata=inF, bd_id=0, mem=F,
                                   sizes=[1, 1, 1, FEATURE_COUNT])
                npu_dma_memcpy_nd(metadata=inW, bd_id=1, mem=W,
                                   sizes=[1, 1, 1, FEATURE_COUNT])
                npu_dma_memcpy_nd(metadata=finalS, bd_id=2, mem=S,
                                   sizes=[1, 1, 1, 1])
                dma_wait(finalS)

    print(ctx.module)


if __name__ == "__main__":
    dotproduct_kernel()