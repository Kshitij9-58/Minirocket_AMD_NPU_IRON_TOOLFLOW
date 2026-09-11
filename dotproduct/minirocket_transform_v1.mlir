module {
  aie.device(npu1) {
    func.func private @zero_i32_scalar(memref<1xi32>)
    func.func private @minirocket_transform_tile(memref<251xi32>, memref<420xi32>, memref<420xi32>, memref<420xi32>, memref<2520xi32>, i32, memref<420xi32>)

    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)

    %series_local_0 = aie.buffer(%tile_0_2) {sym_name = "series_local_0"} : memref<251xi32>
    %bias_local_0 = aie.buffer(%tile_0_2) {sym_name = "bias_local_0"} : memref<420xi32>
    %smean_local_0 = aie.buffer(%tile_0_2) {sym_name = "smean_local_0"} : memref<420xi32>
    %sscale_local_0 = aie.buffer(%tile_0_2) {sym_name = "sscale_local_0"} : memref<420xi32>
    %params_local_0 = aie.buffer(%tile_0_2) {sym_name = "params_local_0"} : memref<2520xi32>
    %series_local_1 = aie.buffer(%tile_1_2) {sym_name = "series_local_1"} : memref<251xi32>
    %bias_local_1 = aie.buffer(%tile_1_2) {sym_name = "bias_local_1"} : memref<420xi32>
    %smean_local_1 = aie.buffer(%tile_1_2) {sym_name = "smean_local_1"} : memref<420xi32>
    %sscale_local_1 = aie.buffer(%tile_1_2) {sym_name = "sscale_local_1"} : memref<420xi32>
    %params_local_1 = aie.buffer(%tile_1_2) {sym_name = "params_local_1"} : memref<2520xi32>

    // --- ROUTING: COLUMN 0 (transform tile 0) ---
    aie.objectfifo @in_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4031xi32>>
    aie.objectfifo @in_mem_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<4031xi32>>
    aie.objectfifo.link [@in_0] -> [@in_mem_0]([] [])
    aie.objectfifo @out_mem_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo @out_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo.link [@out_mem_0] -> [@out_0]([] [])

    // --- ROUTING: COLUMN 1 (transform tile 1) ---
    aie.objectfifo @in_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<4031xi32>>
    aie.objectfifo @in_mem_1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<4031xi32>>
    aie.objectfifo.link [@in_1] -> [@in_mem_1]([] [])
    aie.objectfifo @out_mem_1(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo @out_1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo.link [@out_mem_1] -> [@out_1]([] [])

    // --- CORES ---
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c_series_off = arith.constant 0 : index
      %c_bias_off = arith.constant 251 : index
      %c_smean_off = arith.constant 671 : index
      %c_sscale_off = arith.constant 1091 : index
      %c_params_off = arith.constant 1511 : index
      %c_seq_len = arith.constant 251 : index
      %c_num_feat = arith.constant 420 : i32

      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %in_elem = aie.objectfifo.acquire @in_mem_0(Consume, 1) : !aie.objectfifosubview<memref<4031xi32>>
        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<4031xi32>> -> memref<4031xi32>
        %out = aie.objectfifo.acquire @out_mem_0(Produce, 1) : !aie.objectfifosubview<memref<420xi32>>
        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<420xi32>> -> memref<420xi32>

        scf.for %s_i = %c0 to %c_seq_len step %c1 {
          %s_src_idx = arith.addi %c_series_off, %s_i : index
          %s_val = memref.load %in_v[%s_src_idx] : memref<4031xi32>
          memref.store %s_val, %series_local_0[%s_i] : memref<251xi32>
        }

        %c_F = arith.constant 420 : index
        scf.for %b_i = %c0 to %c_F step %c1 {
          %b_src_idx = arith.addi %c_bias_off, %b_i : index
          %b_val = memref.load %in_v[%b_src_idx] : memref<4031xi32>
          memref.store %b_val, %bias_local_0[%b_i] : memref<420xi32>
          %sm_src_idx = arith.addi %c_smean_off, %b_i : index
          %sm_val = memref.load %in_v[%sm_src_idx] : memref<4031xi32>
          memref.store %sm_val, %smean_local_0[%b_i] : memref<420xi32>
          %ss_src_idx = arith.addi %c_sscale_off, %b_i : index
          %ss_val = memref.load %in_v[%ss_src_idx] : memref<4031xi32>
          memref.store %ss_val, %sscale_local_0[%b_i] : memref<420xi32>
        }

        %c_Pn = arith.constant 2520 : index
        scf.for %p_i = %c0 to %c_Pn step %c1 {
          %p_src_idx = arith.addi %c_params_off, %p_i : index
          %p_val = memref.load %in_v[%p_src_idx] : memref<4031xi32>
          memref.store %p_val, %params_local_0[%p_i] : memref<2520xi32>
        }

        func.call @minirocket_transform_tile(%series_local_0, %bias_local_0, %smean_local_0, %sscale_local_0, %params_local_0, %c_num_feat, %out_v) : (memref<251xi32>, memref<420xi32>, memref<420xi32>, memref<420xi32>, memref<2520xi32>, i32, memref<420xi32>) -> ()

        aie.objectfifo.release @in_mem_0(Consume, 1)
        aie.objectfifo.release @out_mem_0(Produce, 1)
      }
      aie.end
    } {stack_size = 4096 : i32, link_with = "minirocket_transform_kernel.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c_series_off = arith.constant 0 : index
      %c_bias_off = arith.constant 251 : index
      %c_smean_off = arith.constant 671 : index
      %c_sscale_off = arith.constant 1091 : index
      %c_params_off = arith.constant 1511 : index
      %c_seq_len = arith.constant 251 : index
      %c_num_feat = arith.constant 420 : i32

      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %in_elem = aie.objectfifo.acquire @in_mem_1(Consume, 1) : !aie.objectfifosubview<memref<4031xi32>>
        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<4031xi32>> -> memref<4031xi32>
        %out = aie.objectfifo.acquire @out_mem_1(Produce, 1) : !aie.objectfifosubview<memref<420xi32>>
        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<420xi32>> -> memref<420xi32>

        scf.for %s_i = %c0 to %c_seq_len step %c1 {
          %s_src_idx = arith.addi %c_series_off, %s_i : index
          %s_val = memref.load %in_v[%s_src_idx] : memref<4031xi32>
          memref.store %s_val, %series_local_1[%s_i] : memref<251xi32>
        }

        %c_F = arith.constant 420 : index
        scf.for %b_i = %c0 to %c_F step %c1 {
          %b_src_idx = arith.addi %c_bias_off, %b_i : index
          %b_val = memref.load %in_v[%b_src_idx] : memref<4031xi32>
          memref.store %b_val, %bias_local_1[%b_i] : memref<420xi32>
          %sm_src_idx = arith.addi %c_smean_off, %b_i : index
          %sm_val = memref.load %in_v[%sm_src_idx] : memref<4031xi32>
          memref.store %sm_val, %smean_local_1[%b_i] : memref<420xi32>
          %ss_src_idx = arith.addi %c_sscale_off, %b_i : index
          %ss_val = memref.load %in_v[%ss_src_idx] : memref<4031xi32>
          memref.store %ss_val, %sscale_local_1[%b_i] : memref<420xi32>
        }

        %c_Pn = arith.constant 2520 : index
        scf.for %p_i = %c0 to %c_Pn step %c1 {
          %p_src_idx = arith.addi %c_params_off, %p_i : index
          %p_val = memref.load %in_v[%p_src_idx] : memref<4031xi32>
          memref.store %p_val, %params_local_1[%p_i] : memref<2520xi32>
        }

        func.call @minirocket_transform_tile(%series_local_1, %bias_local_1, %smean_local_1, %sscale_local_1, %params_local_1, %c_num_feat, %out_v) : (memref<251xi32>, memref<420xi32>, memref<420xi32>, memref<420xi32>, memref<2520xi32>, i32, memref<420xi32>) -> ()

        aie.objectfifo.release @in_mem_1(Consume, 1)
        aie.objectfifo.release @out_mem_1(Produce, 1)
      }
      aie.end
    } {stack_size = 4096 : i32, link_with = "minirocket_transform_kernel.o"}

    // --- HOST RUNTIME SEQUENCE ---
    aie.runtime_sequence(%arg0: memref<8062xi32>, %arg1: memref<840xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 4031][0, 0, 0, 1]) {id = 0 : i64, metadata = @in_0} : memref<8062xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 420][0, 0, 0, 1]) {id = 1 : i64, metadata = @out_0} : memref<840xi32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 4031][1, 1, 1, 4031][0, 0, 0, 1]) {id = 2 : i64, metadata = @in_1} : memref<8062xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 420][1, 1, 1, 420][0, 0, 0, 1]) {id = 3 : i64, metadata = @out_1} : memref<840xi32>
      aiex.npu.dma_wait {symbol = @out_0}
      aiex.npu.dma_wait {symbol = @out_1}
    }
  }
}