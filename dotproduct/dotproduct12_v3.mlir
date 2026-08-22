module {
  aie.device(npu1) {
    func.func private @zero_f32_scalar(memref<1xf32>)
    func.func private @dotproduct_partial_bf16_f32(memref<70xbf16>, memref<70xbf16>, memref<1xf32>)

    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_5 = aie.tile(1, 5)
    %shim_noc_tile_2_0 = aie.tile(2, 0)
    %mem_tile_2_1 = aie.tile(2, 1)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)

    // --- ROUTING: COLUMN 0 ---
    aie.objectfifo @inF_col0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<280xbf16>>
    aie.objectfifo @memF_col0(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<70xbf16>>
    aie.objectfifo.link [@inF_col0] -> [@memF_col0]([] [])
    aie.objectfifo @inW_col0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<280xbf16>>
    aie.objectfifo @memW_col0(%mem_tile_0_1, {%tile_0_2, %tile_0_3, %tile_0_4, %tile_0_5}, 2 : i32) : !aie.objectfifo<memref<70xbf16>>
    aie.objectfifo.link [@inW_col0] -> [@memW_col0]([] [])
    aie.objectfifo @memC_0_2(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_0_3(%tile_0_3, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_0_4(%tile_0_4, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_0_5(%tile_0_5, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @outC_col0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4xf32>>
    aie.objectfifo.link [@memC_0_2, @memC_0_3, @memC_0_4, @memC_0_5] -> [@outC_col0]([0, 0, 0, 0] [])

    // --- ROUTING: COLUMN 1 ---
    aie.objectfifo @inF_col1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<280xbf16>>
    aie.objectfifo @memF_col1(%mem_tile_1_1, {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<70xbf16>>
    aie.objectfifo.link [@inF_col1] -> [@memF_col1]([] [])
    aie.objectfifo @inW_col1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<280xbf16>>
    aie.objectfifo @memW_col1(%mem_tile_1_1, {%tile_1_2, %tile_1_3, %tile_1_4, %tile_1_5}, 2 : i32) : !aie.objectfifo<memref<70xbf16>>
    aie.objectfifo.link [@inW_col1] -> [@memW_col1]([] [])
    aie.objectfifo @memC_1_2(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_1_3(%tile_1_3, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_1_4(%tile_1_4, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_1_5(%tile_1_5, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @outC_col1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<4xf32>>
    aie.objectfifo.link [@memC_1_2, @memC_1_3, @memC_1_4, @memC_1_5] -> [@outC_col1]([0, 0, 0, 0] [])

    // --- ROUTING: COLUMN 2 ---
    aie.objectfifo @inF_col2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<280xbf16>>
    aie.objectfifo @memF_col2(%mem_tile_2_1, {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<70xbf16>>
    aie.objectfifo.link [@inF_col2] -> [@memF_col2]([] [])
    aie.objectfifo @inW_col2(%shim_noc_tile_2_0, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<280xbf16>>
    aie.objectfifo @memW_col2(%mem_tile_2_1, {%tile_2_2, %tile_2_3, %tile_2_4, %tile_2_5}, 2 : i32) : !aie.objectfifo<memref<70xbf16>>
    aie.objectfifo.link [@inW_col2] -> [@memW_col2]([] [])
    aie.objectfifo @memC_2_2(%tile_2_2, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_2_3(%tile_2_3, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_2_4(%tile_2_4, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @memC_2_5(%tile_2_5, {%mem_tile_2_1}, 2 : i32) : !aie.objectfifo<memref<1xf32>>
    aie.objectfifo @outC_col2(%mem_tile_2_1, {%shim_noc_tile_2_0}, 2 : i32) : !aie.objectfifo<memref<4xf32>>
    aie.objectfifo.link [@memC_2_2, @memC_2_3, @memC_2_4, @memC_2_5] -> [@outC_col2]([0, 0, 0, 0] [])

    // --- CORES ---
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_0_2(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col0(Consume, 1)
        aie.objectfifo.release @memW_col0(Consume, 1)
        aie.objectfifo.release @memC_0_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_0_3(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col0(Consume, 1)
        aie.objectfifo.release @memW_col0(Consume, 1)
        aie.objectfifo.release @memC_0_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_0_4(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col0(Consume, 1)
        aie.objectfifo.release @memW_col0(Consume, 1)
        aie.objectfifo.release @memC_0_4(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_0_5(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col0(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col0(Consume, 1)
        aie.objectfifo.release @memW_col0(Consume, 1)
        aie.objectfifo.release @memC_0_5(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_1_2(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col1(Consume, 1)
        aie.objectfifo.release @memW_col1(Consume, 1)
        aie.objectfifo.release @memC_1_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_1_3(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col1(Consume, 1)
        aie.objectfifo.release @memW_col1(Consume, 1)
        aie.objectfifo.release @memC_1_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_1_4(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col1(Consume, 1)
        aie.objectfifo.release @memW_col1(Consume, 1)
        aie.objectfifo.release @memC_1_4(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_1_5(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col1(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col1(Consume, 1)
        aie.objectfifo.release @memW_col1(Consume, 1)
        aie.objectfifo.release @memC_1_5(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_2_2(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col2(Consume, 1)
        aie.objectfifo.release @memW_col2(Consume, 1)
        aie.objectfifo.release @memC_2_2(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_2_3(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col2(Consume, 1)
        aie.objectfifo.release @memW_col2(Consume, 1)
        aie.objectfifo.release @memC_2_3(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_2_4(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col2(Consume, 1)
        aie.objectfifo.release @memW_col2(Consume, 1)
        aie.objectfifo.release @memC_2_4(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %0 = aie.objectfifo.acquire @memC_2_5(Produce, 1) : !aie.objectfifosubview<memref<1xf32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xf32>> -> memref<1xf32>
        func.call @zero_f32_scalar(%1) : (memref<1xf32>) -> ()
        %2 = aie.objectfifo.acquire @memF_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        %4 = aie.objectfifo.acquire @memW_col2(Consume, 1) : !aie.objectfifosubview<memref<70xbf16>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<70xbf16>> -> memref<70xbf16>
        func.call @dotproduct_partial_bf16_f32(%3, %5, %1) : (memref<70xbf16>, memref<70xbf16>, memref<1xf32>) -> ()
        aie.objectfifo.release @memF_col2(Consume, 1)
        aie.objectfifo.release @memW_col2(Consume, 1)
        aie.objectfifo.release @memC_2_5(Produce, 1)
      }
      aie.end
    } {stack_size = 2048 : i32}

    // --- HOST RUNTIME SEQUENCE ---
    aie.runtime_sequence(%arg0: memref<840xbf16>, %arg1: memref<840xbf16>, %arg2: memref<12xf32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 280][0, 0, 0, 1]) {id = 0 : i64, metadata = @inF_col0} : memref<840xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 280][0, 0, 0, 1]) {id = 1 : i64, metadata = @inW_col0} : memref<840xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 0][1, 1, 1, 4][0, 0, 0, 1]) {id = 2 : i64, metadata = @outC_col0} : memref<12xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 280][1, 1, 1, 280][0, 0, 0, 1]) {id = 3 : i64, metadata = @inF_col1} : memref<840xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 280][1, 1, 1, 280][0, 0, 0, 1]) {id = 4 : i64, metadata = @inW_col1} : memref<840xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 4][1, 1, 1, 4][0, 0, 0, 1]) {id = 5 : i64, metadata = @outC_col1} : memref<12xf32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 560][1, 1, 1, 280][0, 0, 0, 1]) {id = 6 : i64, metadata = @inF_col2} : memref<840xbf16>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 560][1, 1, 1, 280][0, 0, 0, 1]) {id = 7 : i64, metadata = @inW_col2} : memref<840xbf16>
      aiex.npu.dma_memcpy_nd(%arg2[0, 0, 0, 8][1, 1, 1, 4][0, 0, 0, 1]) {id = 8 : i64, metadata = @outC_col2} : memref<12xf32>
      aiex.npu.dma_wait {symbol = @outC_col0}
      aiex.npu.dma_wait {symbol = @outC_col1}
      aiex.npu.dma_wait {symbol = @outC_col2}
    }
  }
}