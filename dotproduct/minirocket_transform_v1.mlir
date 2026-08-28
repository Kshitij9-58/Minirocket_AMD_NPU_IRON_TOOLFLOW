module {
  aie.device(npu1) {
    func.func private @zero_i32_scalar(memref<1xi32>)
    func.func private @minirocket_feature_transform(memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32)

    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)

    // --- ROUTING: COLUMN 0 (transform tile 0) ---
    aie.objectfifo @in_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1511xi32>>
    aie.objectfifo @in_mem_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1511xi32>>
    aie.objectfifo.link [@in_0] -> [@in_mem_0]([] [])
    aie.objectfifo @out_mem_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo @out_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo.link [@out_mem_0] -> [@out_0]([] [])

    // --- ROUTING: COLUMN 1 (transform tile 1) ---
    aie.objectfifo @in_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<1511xi32>>
    aie.objectfifo @in_mem_1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<1511xi32>>
    aie.objectfifo.link [@in_1] -> [@in_mem_1]([] [])
    aie.objectfifo @out_mem_1(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo @out_1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<420xi32>>
    aie.objectfifo.link [@out_mem_1] -> [@out_1]([] [])

    // --- CORES ---
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c_series_off = arith.constant 0 : i32

      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %in_elem = aie.objectfifo.acquire @in_mem_0(Consume, 1) : !aie.objectfifosubview<memref<1511xi32>>
        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<1511xi32>> -> memref<1511xi32>
        %out = aie.objectfifo.acquire @out_mem_0(Produce, 1) : !aie.objectfifosubview<memref<420xi32>>
        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<420xi32>> -> memref<420xi32>

        // feature 0: dilation=1 padding=4 kernel=(0,1,2) padding1=0
        %fidx_0 = arith.constant 0 : index
        %bidx_0 = arith.constant 251 : index
        %smidx_0 = arith.constant 671 : index
        %ssidx_0 = arith.constant 1091 : index
        %bias_0 = memref.load %in_v[%bidx_0] : memref<1511xi32>
        %smean_0 = memref.load %in_v[%smidx_0] : memref<1511xi32>
        %sscale_0 = memref.load %in_v[%ssidx_0] : memref<1511xi32>
        %dilation_0 = arith.constant 1 : i32
        %padding_0 = arith.constant 4 : i32
        %ki0_0 = arith.constant 0 : i32
        %ki1_0 = arith.constant 1 : i32
        %ki2_0 = arith.constant 2 : i32
        %pad1_0 = arith.constant 0 : i32
        %outidx_0 = arith.constant 0 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_0, %smean_0, %sscale_0, %dilation_0, %padding_0, %ki0_0, %ki1_0, %ki2_0, %pad1_0, %out_v, %outidx_0) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 1: dilation=1 padding=4 kernel=(0,1,2) padding1=0
        %fidx_1 = arith.constant 1 : index
        %bidx_1 = arith.constant 252 : index
        %smidx_1 = arith.constant 672 : index
        %ssidx_1 = arith.constant 1092 : index
        %bias_1 = memref.load %in_v[%bidx_1] : memref<1511xi32>
        %smean_1 = memref.load %in_v[%smidx_1] : memref<1511xi32>
        %sscale_1 = memref.load %in_v[%ssidx_1] : memref<1511xi32>
        %dilation_1 = arith.constant 1 : i32
        %padding_1 = arith.constant 4 : i32
        %ki0_1 = arith.constant 0 : i32
        %ki1_1 = arith.constant 1 : i32
        %ki2_1 = arith.constant 2 : i32
        %pad1_1 = arith.constant 0 : i32
        %outidx_1 = arith.constant 1 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_1, %smean_1, %sscale_1, %dilation_1, %padding_1, %ki0_1, %ki1_1, %ki2_1, %pad1_1, %out_v, %outidx_1) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 2: dilation=1 padding=4 kernel=(0,1,3) padding1=1
        %fidx_2 = arith.constant 2 : index
        %bidx_2 = arith.constant 253 : index
        %smidx_2 = arith.constant 673 : index
        %ssidx_2 = arith.constant 1093 : index
        %bias_2 = memref.load %in_v[%bidx_2] : memref<1511xi32>
        %smean_2 = memref.load %in_v[%smidx_2] : memref<1511xi32>
        %sscale_2 = memref.load %in_v[%ssidx_2] : memref<1511xi32>
        %dilation_2 = arith.constant 1 : i32
        %padding_2 = arith.constant 4 : i32
        %ki0_2 = arith.constant 0 : i32
        %ki1_2 = arith.constant 1 : i32
        %ki2_2 = arith.constant 3 : i32
        %pad1_2 = arith.constant 1 : i32
        %outidx_2 = arith.constant 2 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_2, %smean_2, %sscale_2, %dilation_2, %padding_2, %ki0_2, %ki1_2, %ki2_2, %pad1_2, %out_v, %outidx_2) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 3: dilation=1 padding=4 kernel=(0,1,3) padding1=1
        %fidx_3 = arith.constant 3 : index
        %bidx_3 = arith.constant 254 : index
        %smidx_3 = arith.constant 674 : index
        %ssidx_3 = arith.constant 1094 : index
        %bias_3 = memref.load %in_v[%bidx_3] : memref<1511xi32>
        %smean_3 = memref.load %in_v[%smidx_3] : memref<1511xi32>
        %sscale_3 = memref.load %in_v[%ssidx_3] : memref<1511xi32>
        %dilation_3 = arith.constant 1 : i32
        %padding_3 = arith.constant 4 : i32
        %ki0_3 = arith.constant 0 : i32
        %ki1_3 = arith.constant 1 : i32
        %ki2_3 = arith.constant 3 : i32
        %pad1_3 = arith.constant 1 : i32
        %outidx_3 = arith.constant 3 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_3, %smean_3, %sscale_3, %dilation_3, %padding_3, %ki0_3, %ki1_3, %ki2_3, %pad1_3, %out_v, %outidx_3) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 4: dilation=1 padding=4 kernel=(0,1,4) padding1=0
        %fidx_4 = arith.constant 4 : index
        %bidx_4 = arith.constant 255 : index
        %smidx_4 = arith.constant 675 : index
        %ssidx_4 = arith.constant 1095 : index
        %bias_4 = memref.load %in_v[%bidx_4] : memref<1511xi32>
        %smean_4 = memref.load %in_v[%smidx_4] : memref<1511xi32>
        %sscale_4 = memref.load %in_v[%ssidx_4] : memref<1511xi32>
        %dilation_4 = arith.constant 1 : i32
        %padding_4 = arith.constant 4 : i32
        %ki0_4 = arith.constant 0 : i32
        %ki1_4 = arith.constant 1 : i32
        %ki2_4 = arith.constant 4 : i32
        %pad1_4 = arith.constant 0 : i32
        %outidx_4 = arith.constant 4 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_4, %smean_4, %sscale_4, %dilation_4, %padding_4, %ki0_4, %ki1_4, %ki2_4, %pad1_4, %out_v, %outidx_4) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 5: dilation=1 padding=4 kernel=(0,1,4) padding1=0
        %fidx_5 = arith.constant 5 : index
        %bidx_5 = arith.constant 256 : index
        %smidx_5 = arith.constant 676 : index
        %ssidx_5 = arith.constant 1096 : index
        %bias_5 = memref.load %in_v[%bidx_5] : memref<1511xi32>
        %smean_5 = memref.load %in_v[%smidx_5] : memref<1511xi32>
        %sscale_5 = memref.load %in_v[%ssidx_5] : memref<1511xi32>
        %dilation_5 = arith.constant 1 : i32
        %padding_5 = arith.constant 4 : i32
        %ki0_5 = arith.constant 0 : i32
        %ki1_5 = arith.constant 1 : i32
        %ki2_5 = arith.constant 4 : i32
        %pad1_5 = arith.constant 0 : i32
        %outidx_5 = arith.constant 5 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_5, %smean_5, %sscale_5, %dilation_5, %padding_5, %ki0_5, %ki1_5, %ki2_5, %pad1_5, %out_v, %outidx_5) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 6: dilation=1 padding=4 kernel=(0,1,5) padding1=1
        %fidx_6 = arith.constant 6 : index
        %bidx_6 = arith.constant 257 : index
        %smidx_6 = arith.constant 677 : index
        %ssidx_6 = arith.constant 1097 : index
        %bias_6 = memref.load %in_v[%bidx_6] : memref<1511xi32>
        %smean_6 = memref.load %in_v[%smidx_6] : memref<1511xi32>
        %sscale_6 = memref.load %in_v[%ssidx_6] : memref<1511xi32>
        %dilation_6 = arith.constant 1 : i32
        %padding_6 = arith.constant 4 : i32
        %ki0_6 = arith.constant 0 : i32
        %ki1_6 = arith.constant 1 : i32
        %ki2_6 = arith.constant 5 : i32
        %pad1_6 = arith.constant 1 : i32
        %outidx_6 = arith.constant 6 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_6, %smean_6, %sscale_6, %dilation_6, %padding_6, %ki0_6, %ki1_6, %ki2_6, %pad1_6, %out_v, %outidx_6) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 7: dilation=1 padding=4 kernel=(0,1,5) padding1=1
        %fidx_7 = arith.constant 7 : index
        %bidx_7 = arith.constant 258 : index
        %smidx_7 = arith.constant 678 : index
        %ssidx_7 = arith.constant 1098 : index
        %bias_7 = memref.load %in_v[%bidx_7] : memref<1511xi32>
        %smean_7 = memref.load %in_v[%smidx_7] : memref<1511xi32>
        %sscale_7 = memref.load %in_v[%ssidx_7] : memref<1511xi32>
        %dilation_7 = arith.constant 1 : i32
        %padding_7 = arith.constant 4 : i32
        %ki0_7 = arith.constant 0 : i32
        %ki1_7 = arith.constant 1 : i32
        %ki2_7 = arith.constant 5 : i32
        %pad1_7 = arith.constant 1 : i32
        %outidx_7 = arith.constant 7 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_7, %smean_7, %sscale_7, %dilation_7, %padding_7, %ki0_7, %ki1_7, %ki2_7, %pad1_7, %out_v, %outidx_7) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 8: dilation=1 padding=4 kernel=(0,1,6) padding1=0
        %fidx_8 = arith.constant 8 : index
        %bidx_8 = arith.constant 259 : index
        %smidx_8 = arith.constant 679 : index
        %ssidx_8 = arith.constant 1099 : index
        %bias_8 = memref.load %in_v[%bidx_8] : memref<1511xi32>
        %smean_8 = memref.load %in_v[%smidx_8] : memref<1511xi32>
        %sscale_8 = memref.load %in_v[%ssidx_8] : memref<1511xi32>
        %dilation_8 = arith.constant 1 : i32
        %padding_8 = arith.constant 4 : i32
        %ki0_8 = arith.constant 0 : i32
        %ki1_8 = arith.constant 1 : i32
        %ki2_8 = arith.constant 6 : i32
        %pad1_8 = arith.constant 0 : i32
        %outidx_8 = arith.constant 8 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_8, %smean_8, %sscale_8, %dilation_8, %padding_8, %ki0_8, %ki1_8, %ki2_8, %pad1_8, %out_v, %outidx_8) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 9: dilation=1 padding=4 kernel=(0,1,6) padding1=0
        %fidx_9 = arith.constant 9 : index
        %bidx_9 = arith.constant 260 : index
        %smidx_9 = arith.constant 680 : index
        %ssidx_9 = arith.constant 1100 : index
        %bias_9 = memref.load %in_v[%bidx_9] : memref<1511xi32>
        %smean_9 = memref.load %in_v[%smidx_9] : memref<1511xi32>
        %sscale_9 = memref.load %in_v[%ssidx_9] : memref<1511xi32>
        %dilation_9 = arith.constant 1 : i32
        %padding_9 = arith.constant 4 : i32
        %ki0_9 = arith.constant 0 : i32
        %ki1_9 = arith.constant 1 : i32
        %ki2_9 = arith.constant 6 : i32
        %pad1_9 = arith.constant 0 : i32
        %outidx_9 = arith.constant 9 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_9, %smean_9, %sscale_9, %dilation_9, %padding_9, %ki0_9, %ki1_9, %ki2_9, %pad1_9, %out_v, %outidx_9) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 10: dilation=1 padding=4 kernel=(0,1,7) padding1=1
        %fidx_10 = arith.constant 10 : index
        %bidx_10 = arith.constant 261 : index
        %smidx_10 = arith.constant 681 : index
        %ssidx_10 = arith.constant 1101 : index
        %bias_10 = memref.load %in_v[%bidx_10] : memref<1511xi32>
        %smean_10 = memref.load %in_v[%smidx_10] : memref<1511xi32>
        %sscale_10 = memref.load %in_v[%ssidx_10] : memref<1511xi32>
        %dilation_10 = arith.constant 1 : i32
        %padding_10 = arith.constant 4 : i32
        %ki0_10 = arith.constant 0 : i32
        %ki1_10 = arith.constant 1 : i32
        %ki2_10 = arith.constant 7 : i32
        %pad1_10 = arith.constant 1 : i32
        %outidx_10 = arith.constant 10 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_10, %smean_10, %sscale_10, %dilation_10, %padding_10, %ki0_10, %ki1_10, %ki2_10, %pad1_10, %out_v, %outidx_10) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 11: dilation=1 padding=4 kernel=(0,1,7) padding1=1
        %fidx_11 = arith.constant 11 : index
        %bidx_11 = arith.constant 262 : index
        %smidx_11 = arith.constant 682 : index
        %ssidx_11 = arith.constant 1102 : index
        %bias_11 = memref.load %in_v[%bidx_11] : memref<1511xi32>
        %smean_11 = memref.load %in_v[%smidx_11] : memref<1511xi32>
        %sscale_11 = memref.load %in_v[%ssidx_11] : memref<1511xi32>
        %dilation_11 = arith.constant 1 : i32
        %padding_11 = arith.constant 4 : i32
        %ki0_11 = arith.constant 0 : i32
        %ki1_11 = arith.constant 1 : i32
        %ki2_11 = arith.constant 7 : i32
        %pad1_11 = arith.constant 1 : i32
        %outidx_11 = arith.constant 11 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_11, %smean_11, %sscale_11, %dilation_11, %padding_11, %ki0_11, %ki1_11, %ki2_11, %pad1_11, %out_v, %outidx_11) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 12: dilation=1 padding=4 kernel=(0,1,8) padding1=0
        %fidx_12 = arith.constant 12 : index
        %bidx_12 = arith.constant 263 : index
        %smidx_12 = arith.constant 683 : index
        %ssidx_12 = arith.constant 1103 : index
        %bias_12 = memref.load %in_v[%bidx_12] : memref<1511xi32>
        %smean_12 = memref.load %in_v[%smidx_12] : memref<1511xi32>
        %sscale_12 = memref.load %in_v[%ssidx_12] : memref<1511xi32>
        %dilation_12 = arith.constant 1 : i32
        %padding_12 = arith.constant 4 : i32
        %ki0_12 = arith.constant 0 : i32
        %ki1_12 = arith.constant 1 : i32
        %ki2_12 = arith.constant 8 : i32
        %pad1_12 = arith.constant 0 : i32
        %outidx_12 = arith.constant 12 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_12, %smean_12, %sscale_12, %dilation_12, %padding_12, %ki0_12, %ki1_12, %ki2_12, %pad1_12, %out_v, %outidx_12) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 13: dilation=1 padding=4 kernel=(0,1,8) padding1=0
        %fidx_13 = arith.constant 13 : index
        %bidx_13 = arith.constant 264 : index
        %smidx_13 = arith.constant 684 : index
        %ssidx_13 = arith.constant 1104 : index
        %bias_13 = memref.load %in_v[%bidx_13] : memref<1511xi32>
        %smean_13 = memref.load %in_v[%smidx_13] : memref<1511xi32>
        %sscale_13 = memref.load %in_v[%ssidx_13] : memref<1511xi32>
        %dilation_13 = arith.constant 1 : i32
        %padding_13 = arith.constant 4 : i32
        %ki0_13 = arith.constant 0 : i32
        %ki1_13 = arith.constant 1 : i32
        %ki2_13 = arith.constant 8 : i32
        %pad1_13 = arith.constant 0 : i32
        %outidx_13 = arith.constant 13 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_13, %smean_13, %sscale_13, %dilation_13, %padding_13, %ki0_13, %ki1_13, %ki2_13, %pad1_13, %out_v, %outidx_13) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 14: dilation=1 padding=4 kernel=(0,2,3) padding1=1
        %fidx_14 = arith.constant 14 : index
        %bidx_14 = arith.constant 265 : index
        %smidx_14 = arith.constant 685 : index
        %ssidx_14 = arith.constant 1105 : index
        %bias_14 = memref.load %in_v[%bidx_14] : memref<1511xi32>
        %smean_14 = memref.load %in_v[%smidx_14] : memref<1511xi32>
        %sscale_14 = memref.load %in_v[%ssidx_14] : memref<1511xi32>
        %dilation_14 = arith.constant 1 : i32
        %padding_14 = arith.constant 4 : i32
        %ki0_14 = arith.constant 0 : i32
        %ki1_14 = arith.constant 2 : i32
        %ki2_14 = arith.constant 3 : i32
        %pad1_14 = arith.constant 1 : i32
        %outidx_14 = arith.constant 14 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_14, %smean_14, %sscale_14, %dilation_14, %padding_14, %ki0_14, %ki1_14, %ki2_14, %pad1_14, %out_v, %outidx_14) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 15: dilation=1 padding=4 kernel=(0,2,3) padding1=1
        %fidx_15 = arith.constant 15 : index
        %bidx_15 = arith.constant 266 : index
        %smidx_15 = arith.constant 686 : index
        %ssidx_15 = arith.constant 1106 : index
        %bias_15 = memref.load %in_v[%bidx_15] : memref<1511xi32>
        %smean_15 = memref.load %in_v[%smidx_15] : memref<1511xi32>
        %sscale_15 = memref.load %in_v[%ssidx_15] : memref<1511xi32>
        %dilation_15 = arith.constant 1 : i32
        %padding_15 = arith.constant 4 : i32
        %ki0_15 = arith.constant 0 : i32
        %ki1_15 = arith.constant 2 : i32
        %ki2_15 = arith.constant 3 : i32
        %pad1_15 = arith.constant 1 : i32
        %outidx_15 = arith.constant 15 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_15, %smean_15, %sscale_15, %dilation_15, %padding_15, %ki0_15, %ki1_15, %ki2_15, %pad1_15, %out_v, %outidx_15) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 16: dilation=1 padding=4 kernel=(0,2,4) padding1=0
        %fidx_16 = arith.constant 16 : index
        %bidx_16 = arith.constant 267 : index
        %smidx_16 = arith.constant 687 : index
        %ssidx_16 = arith.constant 1107 : index
        %bias_16 = memref.load %in_v[%bidx_16] : memref<1511xi32>
        %smean_16 = memref.load %in_v[%smidx_16] : memref<1511xi32>
        %sscale_16 = memref.load %in_v[%ssidx_16] : memref<1511xi32>
        %dilation_16 = arith.constant 1 : i32
        %padding_16 = arith.constant 4 : i32
        %ki0_16 = arith.constant 0 : i32
        %ki1_16 = arith.constant 2 : i32
        %ki2_16 = arith.constant 4 : i32
        %pad1_16 = arith.constant 0 : i32
        %outidx_16 = arith.constant 16 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_16, %smean_16, %sscale_16, %dilation_16, %padding_16, %ki0_16, %ki1_16, %ki2_16, %pad1_16, %out_v, %outidx_16) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 17: dilation=1 padding=4 kernel=(0,2,4) padding1=0
        %fidx_17 = arith.constant 17 : index
        %bidx_17 = arith.constant 268 : index
        %smidx_17 = arith.constant 688 : index
        %ssidx_17 = arith.constant 1108 : index
        %bias_17 = memref.load %in_v[%bidx_17] : memref<1511xi32>
        %smean_17 = memref.load %in_v[%smidx_17] : memref<1511xi32>
        %sscale_17 = memref.load %in_v[%ssidx_17] : memref<1511xi32>
        %dilation_17 = arith.constant 1 : i32
        %padding_17 = arith.constant 4 : i32
        %ki0_17 = arith.constant 0 : i32
        %ki1_17 = arith.constant 2 : i32
        %ki2_17 = arith.constant 4 : i32
        %pad1_17 = arith.constant 0 : i32
        %outidx_17 = arith.constant 17 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_17, %smean_17, %sscale_17, %dilation_17, %padding_17, %ki0_17, %ki1_17, %ki2_17, %pad1_17, %out_v, %outidx_17) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 18: dilation=1 padding=4 kernel=(0,2,5) padding1=1
        %fidx_18 = arith.constant 18 : index
        %bidx_18 = arith.constant 269 : index
        %smidx_18 = arith.constant 689 : index
        %ssidx_18 = arith.constant 1109 : index
        %bias_18 = memref.load %in_v[%bidx_18] : memref<1511xi32>
        %smean_18 = memref.load %in_v[%smidx_18] : memref<1511xi32>
        %sscale_18 = memref.load %in_v[%ssidx_18] : memref<1511xi32>
        %dilation_18 = arith.constant 1 : i32
        %padding_18 = arith.constant 4 : i32
        %ki0_18 = arith.constant 0 : i32
        %ki1_18 = arith.constant 2 : i32
        %ki2_18 = arith.constant 5 : i32
        %pad1_18 = arith.constant 1 : i32
        %outidx_18 = arith.constant 18 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_18, %smean_18, %sscale_18, %dilation_18, %padding_18, %ki0_18, %ki1_18, %ki2_18, %pad1_18, %out_v, %outidx_18) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 19: dilation=1 padding=4 kernel=(0,2,5) padding1=1
        %fidx_19 = arith.constant 19 : index
        %bidx_19 = arith.constant 270 : index
        %smidx_19 = arith.constant 690 : index
        %ssidx_19 = arith.constant 1110 : index
        %bias_19 = memref.load %in_v[%bidx_19] : memref<1511xi32>
        %smean_19 = memref.load %in_v[%smidx_19] : memref<1511xi32>
        %sscale_19 = memref.load %in_v[%ssidx_19] : memref<1511xi32>
        %dilation_19 = arith.constant 1 : i32
        %padding_19 = arith.constant 4 : i32
        %ki0_19 = arith.constant 0 : i32
        %ki1_19 = arith.constant 2 : i32
        %ki2_19 = arith.constant 5 : i32
        %pad1_19 = arith.constant 1 : i32
        %outidx_19 = arith.constant 19 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_19, %smean_19, %sscale_19, %dilation_19, %padding_19, %ki0_19, %ki1_19, %ki2_19, %pad1_19, %out_v, %outidx_19) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 20: dilation=1 padding=4 kernel=(0,2,6) padding1=0
        %fidx_20 = arith.constant 20 : index
        %bidx_20 = arith.constant 271 : index
        %smidx_20 = arith.constant 691 : index
        %ssidx_20 = arith.constant 1111 : index
        %bias_20 = memref.load %in_v[%bidx_20] : memref<1511xi32>
        %smean_20 = memref.load %in_v[%smidx_20] : memref<1511xi32>
        %sscale_20 = memref.load %in_v[%ssidx_20] : memref<1511xi32>
        %dilation_20 = arith.constant 1 : i32
        %padding_20 = arith.constant 4 : i32
        %ki0_20 = arith.constant 0 : i32
        %ki1_20 = arith.constant 2 : i32
        %ki2_20 = arith.constant 6 : i32
        %pad1_20 = arith.constant 0 : i32
        %outidx_20 = arith.constant 20 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_20, %smean_20, %sscale_20, %dilation_20, %padding_20, %ki0_20, %ki1_20, %ki2_20, %pad1_20, %out_v, %outidx_20) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 21: dilation=1 padding=4 kernel=(0,2,6) padding1=0
        %fidx_21 = arith.constant 21 : index
        %bidx_21 = arith.constant 272 : index
        %smidx_21 = arith.constant 692 : index
        %ssidx_21 = arith.constant 1112 : index
        %bias_21 = memref.load %in_v[%bidx_21] : memref<1511xi32>
        %smean_21 = memref.load %in_v[%smidx_21] : memref<1511xi32>
        %sscale_21 = memref.load %in_v[%ssidx_21] : memref<1511xi32>
        %dilation_21 = arith.constant 1 : i32
        %padding_21 = arith.constant 4 : i32
        %ki0_21 = arith.constant 0 : i32
        %ki1_21 = arith.constant 2 : i32
        %ki2_21 = arith.constant 6 : i32
        %pad1_21 = arith.constant 0 : i32
        %outidx_21 = arith.constant 21 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_21, %smean_21, %sscale_21, %dilation_21, %padding_21, %ki0_21, %ki1_21, %ki2_21, %pad1_21, %out_v, %outidx_21) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 22: dilation=1 padding=4 kernel=(0,2,7) padding1=1
        %fidx_22 = arith.constant 22 : index
        %bidx_22 = arith.constant 273 : index
        %smidx_22 = arith.constant 693 : index
        %ssidx_22 = arith.constant 1113 : index
        %bias_22 = memref.load %in_v[%bidx_22] : memref<1511xi32>
        %smean_22 = memref.load %in_v[%smidx_22] : memref<1511xi32>
        %sscale_22 = memref.load %in_v[%ssidx_22] : memref<1511xi32>
        %dilation_22 = arith.constant 1 : i32
        %padding_22 = arith.constant 4 : i32
        %ki0_22 = arith.constant 0 : i32
        %ki1_22 = arith.constant 2 : i32
        %ki2_22 = arith.constant 7 : i32
        %pad1_22 = arith.constant 1 : i32
        %outidx_22 = arith.constant 22 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_22, %smean_22, %sscale_22, %dilation_22, %padding_22, %ki0_22, %ki1_22, %ki2_22, %pad1_22, %out_v, %outidx_22) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 23: dilation=1 padding=4 kernel=(0,2,7) padding1=1
        %fidx_23 = arith.constant 23 : index
        %bidx_23 = arith.constant 274 : index
        %smidx_23 = arith.constant 694 : index
        %ssidx_23 = arith.constant 1114 : index
        %bias_23 = memref.load %in_v[%bidx_23] : memref<1511xi32>
        %smean_23 = memref.load %in_v[%smidx_23] : memref<1511xi32>
        %sscale_23 = memref.load %in_v[%ssidx_23] : memref<1511xi32>
        %dilation_23 = arith.constant 1 : i32
        %padding_23 = arith.constant 4 : i32
        %ki0_23 = arith.constant 0 : i32
        %ki1_23 = arith.constant 2 : i32
        %ki2_23 = arith.constant 7 : i32
        %pad1_23 = arith.constant 1 : i32
        %outidx_23 = arith.constant 23 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_23, %smean_23, %sscale_23, %dilation_23, %padding_23, %ki0_23, %ki1_23, %ki2_23, %pad1_23, %out_v, %outidx_23) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 24: dilation=1 padding=4 kernel=(0,2,8) padding1=0
        %fidx_24 = arith.constant 24 : index
        %bidx_24 = arith.constant 275 : index
        %smidx_24 = arith.constant 695 : index
        %ssidx_24 = arith.constant 1115 : index
        %bias_24 = memref.load %in_v[%bidx_24] : memref<1511xi32>
        %smean_24 = memref.load %in_v[%smidx_24] : memref<1511xi32>
        %sscale_24 = memref.load %in_v[%ssidx_24] : memref<1511xi32>
        %dilation_24 = arith.constant 1 : i32
        %padding_24 = arith.constant 4 : i32
        %ki0_24 = arith.constant 0 : i32
        %ki1_24 = arith.constant 2 : i32
        %ki2_24 = arith.constant 8 : i32
        %pad1_24 = arith.constant 0 : i32
        %outidx_24 = arith.constant 24 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_24, %smean_24, %sscale_24, %dilation_24, %padding_24, %ki0_24, %ki1_24, %ki2_24, %pad1_24, %out_v, %outidx_24) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 25: dilation=1 padding=4 kernel=(0,2,8) padding1=0
        %fidx_25 = arith.constant 25 : index
        %bidx_25 = arith.constant 276 : index
        %smidx_25 = arith.constant 696 : index
        %ssidx_25 = arith.constant 1116 : index
        %bias_25 = memref.load %in_v[%bidx_25] : memref<1511xi32>
        %smean_25 = memref.load %in_v[%smidx_25] : memref<1511xi32>
        %sscale_25 = memref.load %in_v[%ssidx_25] : memref<1511xi32>
        %dilation_25 = arith.constant 1 : i32
        %padding_25 = arith.constant 4 : i32
        %ki0_25 = arith.constant 0 : i32
        %ki1_25 = arith.constant 2 : i32
        %ki2_25 = arith.constant 8 : i32
        %pad1_25 = arith.constant 0 : i32
        %outidx_25 = arith.constant 25 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_25, %smean_25, %sscale_25, %dilation_25, %padding_25, %ki0_25, %ki1_25, %ki2_25, %pad1_25, %out_v, %outidx_25) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 26: dilation=1 padding=4 kernel=(0,3,4) padding1=1
        %fidx_26 = arith.constant 26 : index
        %bidx_26 = arith.constant 277 : index
        %smidx_26 = arith.constant 697 : index
        %ssidx_26 = arith.constant 1117 : index
        %bias_26 = memref.load %in_v[%bidx_26] : memref<1511xi32>
        %smean_26 = memref.load %in_v[%smidx_26] : memref<1511xi32>
        %sscale_26 = memref.load %in_v[%ssidx_26] : memref<1511xi32>
        %dilation_26 = arith.constant 1 : i32
        %padding_26 = arith.constant 4 : i32
        %ki0_26 = arith.constant 0 : i32
        %ki1_26 = arith.constant 3 : i32
        %ki2_26 = arith.constant 4 : i32
        %pad1_26 = arith.constant 1 : i32
        %outidx_26 = arith.constant 26 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_26, %smean_26, %sscale_26, %dilation_26, %padding_26, %ki0_26, %ki1_26, %ki2_26, %pad1_26, %out_v, %outidx_26) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 27: dilation=1 padding=4 kernel=(0,3,4) padding1=1
        %fidx_27 = arith.constant 27 : index
        %bidx_27 = arith.constant 278 : index
        %smidx_27 = arith.constant 698 : index
        %ssidx_27 = arith.constant 1118 : index
        %bias_27 = memref.load %in_v[%bidx_27] : memref<1511xi32>
        %smean_27 = memref.load %in_v[%smidx_27] : memref<1511xi32>
        %sscale_27 = memref.load %in_v[%ssidx_27] : memref<1511xi32>
        %dilation_27 = arith.constant 1 : i32
        %padding_27 = arith.constant 4 : i32
        %ki0_27 = arith.constant 0 : i32
        %ki1_27 = arith.constant 3 : i32
        %ki2_27 = arith.constant 4 : i32
        %pad1_27 = arith.constant 1 : i32
        %outidx_27 = arith.constant 27 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_27, %smean_27, %sscale_27, %dilation_27, %padding_27, %ki0_27, %ki1_27, %ki2_27, %pad1_27, %out_v, %outidx_27) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 28: dilation=1 padding=4 kernel=(0,3,5) padding1=0
        %fidx_28 = arith.constant 28 : index
        %bidx_28 = arith.constant 279 : index
        %smidx_28 = arith.constant 699 : index
        %ssidx_28 = arith.constant 1119 : index
        %bias_28 = memref.load %in_v[%bidx_28] : memref<1511xi32>
        %smean_28 = memref.load %in_v[%smidx_28] : memref<1511xi32>
        %sscale_28 = memref.load %in_v[%ssidx_28] : memref<1511xi32>
        %dilation_28 = arith.constant 1 : i32
        %padding_28 = arith.constant 4 : i32
        %ki0_28 = arith.constant 0 : i32
        %ki1_28 = arith.constant 3 : i32
        %ki2_28 = arith.constant 5 : i32
        %pad1_28 = arith.constant 0 : i32
        %outidx_28 = arith.constant 28 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_28, %smean_28, %sscale_28, %dilation_28, %padding_28, %ki0_28, %ki1_28, %ki2_28, %pad1_28, %out_v, %outidx_28) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 29: dilation=1 padding=4 kernel=(0,3,5) padding1=0
        %fidx_29 = arith.constant 29 : index
        %bidx_29 = arith.constant 280 : index
        %smidx_29 = arith.constant 700 : index
        %ssidx_29 = arith.constant 1120 : index
        %bias_29 = memref.load %in_v[%bidx_29] : memref<1511xi32>
        %smean_29 = memref.load %in_v[%smidx_29] : memref<1511xi32>
        %sscale_29 = memref.load %in_v[%ssidx_29] : memref<1511xi32>
        %dilation_29 = arith.constant 1 : i32
        %padding_29 = arith.constant 4 : i32
        %ki0_29 = arith.constant 0 : i32
        %ki1_29 = arith.constant 3 : i32
        %ki2_29 = arith.constant 5 : i32
        %pad1_29 = arith.constant 0 : i32
        %outidx_29 = arith.constant 29 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_29, %smean_29, %sscale_29, %dilation_29, %padding_29, %ki0_29, %ki1_29, %ki2_29, %pad1_29, %out_v, %outidx_29) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 30: dilation=1 padding=4 kernel=(0,3,6) padding1=1
        %fidx_30 = arith.constant 30 : index
        %bidx_30 = arith.constant 281 : index
        %smidx_30 = arith.constant 701 : index
        %ssidx_30 = arith.constant 1121 : index
        %bias_30 = memref.load %in_v[%bidx_30] : memref<1511xi32>
        %smean_30 = memref.load %in_v[%smidx_30] : memref<1511xi32>
        %sscale_30 = memref.load %in_v[%ssidx_30] : memref<1511xi32>
        %dilation_30 = arith.constant 1 : i32
        %padding_30 = arith.constant 4 : i32
        %ki0_30 = arith.constant 0 : i32
        %ki1_30 = arith.constant 3 : i32
        %ki2_30 = arith.constant 6 : i32
        %pad1_30 = arith.constant 1 : i32
        %outidx_30 = arith.constant 30 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_30, %smean_30, %sscale_30, %dilation_30, %padding_30, %ki0_30, %ki1_30, %ki2_30, %pad1_30, %out_v, %outidx_30) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 31: dilation=1 padding=4 kernel=(0,3,6) padding1=1
        %fidx_31 = arith.constant 31 : index
        %bidx_31 = arith.constant 282 : index
        %smidx_31 = arith.constant 702 : index
        %ssidx_31 = arith.constant 1122 : index
        %bias_31 = memref.load %in_v[%bidx_31] : memref<1511xi32>
        %smean_31 = memref.load %in_v[%smidx_31] : memref<1511xi32>
        %sscale_31 = memref.load %in_v[%ssidx_31] : memref<1511xi32>
        %dilation_31 = arith.constant 1 : i32
        %padding_31 = arith.constant 4 : i32
        %ki0_31 = arith.constant 0 : i32
        %ki1_31 = arith.constant 3 : i32
        %ki2_31 = arith.constant 6 : i32
        %pad1_31 = arith.constant 1 : i32
        %outidx_31 = arith.constant 31 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_31, %smean_31, %sscale_31, %dilation_31, %padding_31, %ki0_31, %ki1_31, %ki2_31, %pad1_31, %out_v, %outidx_31) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 32: dilation=1 padding=4 kernel=(0,3,7) padding1=0
        %fidx_32 = arith.constant 32 : index
        %bidx_32 = arith.constant 283 : index
        %smidx_32 = arith.constant 703 : index
        %ssidx_32 = arith.constant 1123 : index
        %bias_32 = memref.load %in_v[%bidx_32] : memref<1511xi32>
        %smean_32 = memref.load %in_v[%smidx_32] : memref<1511xi32>
        %sscale_32 = memref.load %in_v[%ssidx_32] : memref<1511xi32>
        %dilation_32 = arith.constant 1 : i32
        %padding_32 = arith.constant 4 : i32
        %ki0_32 = arith.constant 0 : i32
        %ki1_32 = arith.constant 3 : i32
        %ki2_32 = arith.constant 7 : i32
        %pad1_32 = arith.constant 0 : i32
        %outidx_32 = arith.constant 32 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_32, %smean_32, %sscale_32, %dilation_32, %padding_32, %ki0_32, %ki1_32, %ki2_32, %pad1_32, %out_v, %outidx_32) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 33: dilation=1 padding=4 kernel=(0,3,7) padding1=0
        %fidx_33 = arith.constant 33 : index
        %bidx_33 = arith.constant 284 : index
        %smidx_33 = arith.constant 704 : index
        %ssidx_33 = arith.constant 1124 : index
        %bias_33 = memref.load %in_v[%bidx_33] : memref<1511xi32>
        %smean_33 = memref.load %in_v[%smidx_33] : memref<1511xi32>
        %sscale_33 = memref.load %in_v[%ssidx_33] : memref<1511xi32>
        %dilation_33 = arith.constant 1 : i32
        %padding_33 = arith.constant 4 : i32
        %ki0_33 = arith.constant 0 : i32
        %ki1_33 = arith.constant 3 : i32
        %ki2_33 = arith.constant 7 : i32
        %pad1_33 = arith.constant 0 : i32
        %outidx_33 = arith.constant 33 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_33, %smean_33, %sscale_33, %dilation_33, %padding_33, %ki0_33, %ki1_33, %ki2_33, %pad1_33, %out_v, %outidx_33) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 34: dilation=1 padding=4 kernel=(0,3,8) padding1=1
        %fidx_34 = arith.constant 34 : index
        %bidx_34 = arith.constant 285 : index
        %smidx_34 = arith.constant 705 : index
        %ssidx_34 = arith.constant 1125 : index
        %bias_34 = memref.load %in_v[%bidx_34] : memref<1511xi32>
        %smean_34 = memref.load %in_v[%smidx_34] : memref<1511xi32>
        %sscale_34 = memref.load %in_v[%ssidx_34] : memref<1511xi32>
        %dilation_34 = arith.constant 1 : i32
        %padding_34 = arith.constant 4 : i32
        %ki0_34 = arith.constant 0 : i32
        %ki1_34 = arith.constant 3 : i32
        %ki2_34 = arith.constant 8 : i32
        %pad1_34 = arith.constant 1 : i32
        %outidx_34 = arith.constant 34 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_34, %smean_34, %sscale_34, %dilation_34, %padding_34, %ki0_34, %ki1_34, %ki2_34, %pad1_34, %out_v, %outidx_34) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 35: dilation=1 padding=4 kernel=(0,3,8) padding1=1
        %fidx_35 = arith.constant 35 : index
        %bidx_35 = arith.constant 286 : index
        %smidx_35 = arith.constant 706 : index
        %ssidx_35 = arith.constant 1126 : index
        %bias_35 = memref.load %in_v[%bidx_35] : memref<1511xi32>
        %smean_35 = memref.load %in_v[%smidx_35] : memref<1511xi32>
        %sscale_35 = memref.load %in_v[%ssidx_35] : memref<1511xi32>
        %dilation_35 = arith.constant 1 : i32
        %padding_35 = arith.constant 4 : i32
        %ki0_35 = arith.constant 0 : i32
        %ki1_35 = arith.constant 3 : i32
        %ki2_35 = arith.constant 8 : i32
        %pad1_35 = arith.constant 1 : i32
        %outidx_35 = arith.constant 35 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_35, %smean_35, %sscale_35, %dilation_35, %padding_35, %ki0_35, %ki1_35, %ki2_35, %pad1_35, %out_v, %outidx_35) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 36: dilation=1 padding=4 kernel=(0,4,5) padding1=0
        %fidx_36 = arith.constant 36 : index
        %bidx_36 = arith.constant 287 : index
        %smidx_36 = arith.constant 707 : index
        %ssidx_36 = arith.constant 1127 : index
        %bias_36 = memref.load %in_v[%bidx_36] : memref<1511xi32>
        %smean_36 = memref.load %in_v[%smidx_36] : memref<1511xi32>
        %sscale_36 = memref.load %in_v[%ssidx_36] : memref<1511xi32>
        %dilation_36 = arith.constant 1 : i32
        %padding_36 = arith.constant 4 : i32
        %ki0_36 = arith.constant 0 : i32
        %ki1_36 = arith.constant 4 : i32
        %ki2_36 = arith.constant 5 : i32
        %pad1_36 = arith.constant 0 : i32
        %outidx_36 = arith.constant 36 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_36, %smean_36, %sscale_36, %dilation_36, %padding_36, %ki0_36, %ki1_36, %ki2_36, %pad1_36, %out_v, %outidx_36) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 37: dilation=1 padding=4 kernel=(0,4,5) padding1=0
        %fidx_37 = arith.constant 37 : index
        %bidx_37 = arith.constant 288 : index
        %smidx_37 = arith.constant 708 : index
        %ssidx_37 = arith.constant 1128 : index
        %bias_37 = memref.load %in_v[%bidx_37] : memref<1511xi32>
        %smean_37 = memref.load %in_v[%smidx_37] : memref<1511xi32>
        %sscale_37 = memref.load %in_v[%ssidx_37] : memref<1511xi32>
        %dilation_37 = arith.constant 1 : i32
        %padding_37 = arith.constant 4 : i32
        %ki0_37 = arith.constant 0 : i32
        %ki1_37 = arith.constant 4 : i32
        %ki2_37 = arith.constant 5 : i32
        %pad1_37 = arith.constant 0 : i32
        %outidx_37 = arith.constant 37 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_37, %smean_37, %sscale_37, %dilation_37, %padding_37, %ki0_37, %ki1_37, %ki2_37, %pad1_37, %out_v, %outidx_37) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 38: dilation=1 padding=4 kernel=(0,4,6) padding1=1
        %fidx_38 = arith.constant 38 : index
        %bidx_38 = arith.constant 289 : index
        %smidx_38 = arith.constant 709 : index
        %ssidx_38 = arith.constant 1129 : index
        %bias_38 = memref.load %in_v[%bidx_38] : memref<1511xi32>
        %smean_38 = memref.load %in_v[%smidx_38] : memref<1511xi32>
        %sscale_38 = memref.load %in_v[%ssidx_38] : memref<1511xi32>
        %dilation_38 = arith.constant 1 : i32
        %padding_38 = arith.constant 4 : i32
        %ki0_38 = arith.constant 0 : i32
        %ki1_38 = arith.constant 4 : i32
        %ki2_38 = arith.constant 6 : i32
        %pad1_38 = arith.constant 1 : i32
        %outidx_38 = arith.constant 38 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_38, %smean_38, %sscale_38, %dilation_38, %padding_38, %ki0_38, %ki1_38, %ki2_38, %pad1_38, %out_v, %outidx_38) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 39: dilation=1 padding=4 kernel=(0,4,6) padding1=1
        %fidx_39 = arith.constant 39 : index
        %bidx_39 = arith.constant 290 : index
        %smidx_39 = arith.constant 710 : index
        %ssidx_39 = arith.constant 1130 : index
        %bias_39 = memref.load %in_v[%bidx_39] : memref<1511xi32>
        %smean_39 = memref.load %in_v[%smidx_39] : memref<1511xi32>
        %sscale_39 = memref.load %in_v[%ssidx_39] : memref<1511xi32>
        %dilation_39 = arith.constant 1 : i32
        %padding_39 = arith.constant 4 : i32
        %ki0_39 = arith.constant 0 : i32
        %ki1_39 = arith.constant 4 : i32
        %ki2_39 = arith.constant 6 : i32
        %pad1_39 = arith.constant 1 : i32
        %outidx_39 = arith.constant 39 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_39, %smean_39, %sscale_39, %dilation_39, %padding_39, %ki0_39, %ki1_39, %ki2_39, %pad1_39, %out_v, %outidx_39) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 40: dilation=1 padding=4 kernel=(0,4,7) padding1=0
        %fidx_40 = arith.constant 40 : index
        %bidx_40 = arith.constant 291 : index
        %smidx_40 = arith.constant 711 : index
        %ssidx_40 = arith.constant 1131 : index
        %bias_40 = memref.load %in_v[%bidx_40] : memref<1511xi32>
        %smean_40 = memref.load %in_v[%smidx_40] : memref<1511xi32>
        %sscale_40 = memref.load %in_v[%ssidx_40] : memref<1511xi32>
        %dilation_40 = arith.constant 1 : i32
        %padding_40 = arith.constant 4 : i32
        %ki0_40 = arith.constant 0 : i32
        %ki1_40 = arith.constant 4 : i32
        %ki2_40 = arith.constant 7 : i32
        %pad1_40 = arith.constant 0 : i32
        %outidx_40 = arith.constant 40 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_40, %smean_40, %sscale_40, %dilation_40, %padding_40, %ki0_40, %ki1_40, %ki2_40, %pad1_40, %out_v, %outidx_40) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 41: dilation=1 padding=4 kernel=(0,4,7) padding1=0
        %fidx_41 = arith.constant 41 : index
        %bidx_41 = arith.constant 292 : index
        %smidx_41 = arith.constant 712 : index
        %ssidx_41 = arith.constant 1132 : index
        %bias_41 = memref.load %in_v[%bidx_41] : memref<1511xi32>
        %smean_41 = memref.load %in_v[%smidx_41] : memref<1511xi32>
        %sscale_41 = memref.load %in_v[%ssidx_41] : memref<1511xi32>
        %dilation_41 = arith.constant 1 : i32
        %padding_41 = arith.constant 4 : i32
        %ki0_41 = arith.constant 0 : i32
        %ki1_41 = arith.constant 4 : i32
        %ki2_41 = arith.constant 7 : i32
        %pad1_41 = arith.constant 0 : i32
        %outidx_41 = arith.constant 41 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_41, %smean_41, %sscale_41, %dilation_41, %padding_41, %ki0_41, %ki1_41, %ki2_41, %pad1_41, %out_v, %outidx_41) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 42: dilation=1 padding=4 kernel=(0,4,8) padding1=1
        %fidx_42 = arith.constant 42 : index
        %bidx_42 = arith.constant 293 : index
        %smidx_42 = arith.constant 713 : index
        %ssidx_42 = arith.constant 1133 : index
        %bias_42 = memref.load %in_v[%bidx_42] : memref<1511xi32>
        %smean_42 = memref.load %in_v[%smidx_42] : memref<1511xi32>
        %sscale_42 = memref.load %in_v[%ssidx_42] : memref<1511xi32>
        %dilation_42 = arith.constant 1 : i32
        %padding_42 = arith.constant 4 : i32
        %ki0_42 = arith.constant 0 : i32
        %ki1_42 = arith.constant 4 : i32
        %ki2_42 = arith.constant 8 : i32
        %pad1_42 = arith.constant 1 : i32
        %outidx_42 = arith.constant 42 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_42, %smean_42, %sscale_42, %dilation_42, %padding_42, %ki0_42, %ki1_42, %ki2_42, %pad1_42, %out_v, %outidx_42) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 43: dilation=1 padding=4 kernel=(0,4,8) padding1=1
        %fidx_43 = arith.constant 43 : index
        %bidx_43 = arith.constant 294 : index
        %smidx_43 = arith.constant 714 : index
        %ssidx_43 = arith.constant 1134 : index
        %bias_43 = memref.load %in_v[%bidx_43] : memref<1511xi32>
        %smean_43 = memref.load %in_v[%smidx_43] : memref<1511xi32>
        %sscale_43 = memref.load %in_v[%ssidx_43] : memref<1511xi32>
        %dilation_43 = arith.constant 1 : i32
        %padding_43 = arith.constant 4 : i32
        %ki0_43 = arith.constant 0 : i32
        %ki1_43 = arith.constant 4 : i32
        %ki2_43 = arith.constant 8 : i32
        %pad1_43 = arith.constant 1 : i32
        %outidx_43 = arith.constant 43 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_43, %smean_43, %sscale_43, %dilation_43, %padding_43, %ki0_43, %ki1_43, %ki2_43, %pad1_43, %out_v, %outidx_43) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 44: dilation=1 padding=4 kernel=(0,5,6) padding1=0
        %fidx_44 = arith.constant 44 : index
        %bidx_44 = arith.constant 295 : index
        %smidx_44 = arith.constant 715 : index
        %ssidx_44 = arith.constant 1135 : index
        %bias_44 = memref.load %in_v[%bidx_44] : memref<1511xi32>
        %smean_44 = memref.load %in_v[%smidx_44] : memref<1511xi32>
        %sscale_44 = memref.load %in_v[%ssidx_44] : memref<1511xi32>
        %dilation_44 = arith.constant 1 : i32
        %padding_44 = arith.constant 4 : i32
        %ki0_44 = arith.constant 0 : i32
        %ki1_44 = arith.constant 5 : i32
        %ki2_44 = arith.constant 6 : i32
        %pad1_44 = arith.constant 0 : i32
        %outidx_44 = arith.constant 44 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_44, %smean_44, %sscale_44, %dilation_44, %padding_44, %ki0_44, %ki1_44, %ki2_44, %pad1_44, %out_v, %outidx_44) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 45: dilation=1 padding=4 kernel=(0,5,6) padding1=0
        %fidx_45 = arith.constant 45 : index
        %bidx_45 = arith.constant 296 : index
        %smidx_45 = arith.constant 716 : index
        %ssidx_45 = arith.constant 1136 : index
        %bias_45 = memref.load %in_v[%bidx_45] : memref<1511xi32>
        %smean_45 = memref.load %in_v[%smidx_45] : memref<1511xi32>
        %sscale_45 = memref.load %in_v[%ssidx_45] : memref<1511xi32>
        %dilation_45 = arith.constant 1 : i32
        %padding_45 = arith.constant 4 : i32
        %ki0_45 = arith.constant 0 : i32
        %ki1_45 = arith.constant 5 : i32
        %ki2_45 = arith.constant 6 : i32
        %pad1_45 = arith.constant 0 : i32
        %outidx_45 = arith.constant 45 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_45, %smean_45, %sscale_45, %dilation_45, %padding_45, %ki0_45, %ki1_45, %ki2_45, %pad1_45, %out_v, %outidx_45) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 46: dilation=1 padding=4 kernel=(0,5,7) padding1=1
        %fidx_46 = arith.constant 46 : index
        %bidx_46 = arith.constant 297 : index
        %smidx_46 = arith.constant 717 : index
        %ssidx_46 = arith.constant 1137 : index
        %bias_46 = memref.load %in_v[%bidx_46] : memref<1511xi32>
        %smean_46 = memref.load %in_v[%smidx_46] : memref<1511xi32>
        %sscale_46 = memref.load %in_v[%ssidx_46] : memref<1511xi32>
        %dilation_46 = arith.constant 1 : i32
        %padding_46 = arith.constant 4 : i32
        %ki0_46 = arith.constant 0 : i32
        %ki1_46 = arith.constant 5 : i32
        %ki2_46 = arith.constant 7 : i32
        %pad1_46 = arith.constant 1 : i32
        %outidx_46 = arith.constant 46 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_46, %smean_46, %sscale_46, %dilation_46, %padding_46, %ki0_46, %ki1_46, %ki2_46, %pad1_46, %out_v, %outidx_46) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 47: dilation=1 padding=4 kernel=(0,5,7) padding1=1
        %fidx_47 = arith.constant 47 : index
        %bidx_47 = arith.constant 298 : index
        %smidx_47 = arith.constant 718 : index
        %ssidx_47 = arith.constant 1138 : index
        %bias_47 = memref.load %in_v[%bidx_47] : memref<1511xi32>
        %smean_47 = memref.load %in_v[%smidx_47] : memref<1511xi32>
        %sscale_47 = memref.load %in_v[%ssidx_47] : memref<1511xi32>
        %dilation_47 = arith.constant 1 : i32
        %padding_47 = arith.constant 4 : i32
        %ki0_47 = arith.constant 0 : i32
        %ki1_47 = arith.constant 5 : i32
        %ki2_47 = arith.constant 7 : i32
        %pad1_47 = arith.constant 1 : i32
        %outidx_47 = arith.constant 47 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_47, %smean_47, %sscale_47, %dilation_47, %padding_47, %ki0_47, %ki1_47, %ki2_47, %pad1_47, %out_v, %outidx_47) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 48: dilation=1 padding=4 kernel=(0,5,8) padding1=0
        %fidx_48 = arith.constant 48 : index
        %bidx_48 = arith.constant 299 : index
        %smidx_48 = arith.constant 719 : index
        %ssidx_48 = arith.constant 1139 : index
        %bias_48 = memref.load %in_v[%bidx_48] : memref<1511xi32>
        %smean_48 = memref.load %in_v[%smidx_48] : memref<1511xi32>
        %sscale_48 = memref.load %in_v[%ssidx_48] : memref<1511xi32>
        %dilation_48 = arith.constant 1 : i32
        %padding_48 = arith.constant 4 : i32
        %ki0_48 = arith.constant 0 : i32
        %ki1_48 = arith.constant 5 : i32
        %ki2_48 = arith.constant 8 : i32
        %pad1_48 = arith.constant 0 : i32
        %outidx_48 = arith.constant 48 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_48, %smean_48, %sscale_48, %dilation_48, %padding_48, %ki0_48, %ki1_48, %ki2_48, %pad1_48, %out_v, %outidx_48) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 49: dilation=1 padding=4 kernel=(0,5,8) padding1=0
        %fidx_49 = arith.constant 49 : index
        %bidx_49 = arith.constant 300 : index
        %smidx_49 = arith.constant 720 : index
        %ssidx_49 = arith.constant 1140 : index
        %bias_49 = memref.load %in_v[%bidx_49] : memref<1511xi32>
        %smean_49 = memref.load %in_v[%smidx_49] : memref<1511xi32>
        %sscale_49 = memref.load %in_v[%ssidx_49] : memref<1511xi32>
        %dilation_49 = arith.constant 1 : i32
        %padding_49 = arith.constant 4 : i32
        %ki0_49 = arith.constant 0 : i32
        %ki1_49 = arith.constant 5 : i32
        %ki2_49 = arith.constant 8 : i32
        %pad1_49 = arith.constant 0 : i32
        %outidx_49 = arith.constant 49 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_49, %smean_49, %sscale_49, %dilation_49, %padding_49, %ki0_49, %ki1_49, %ki2_49, %pad1_49, %out_v, %outidx_49) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 50: dilation=1 padding=4 kernel=(0,6,7) padding1=1
        %fidx_50 = arith.constant 50 : index
        %bidx_50 = arith.constant 301 : index
        %smidx_50 = arith.constant 721 : index
        %ssidx_50 = arith.constant 1141 : index
        %bias_50 = memref.load %in_v[%bidx_50] : memref<1511xi32>
        %smean_50 = memref.load %in_v[%smidx_50] : memref<1511xi32>
        %sscale_50 = memref.load %in_v[%ssidx_50] : memref<1511xi32>
        %dilation_50 = arith.constant 1 : i32
        %padding_50 = arith.constant 4 : i32
        %ki0_50 = arith.constant 0 : i32
        %ki1_50 = arith.constant 6 : i32
        %ki2_50 = arith.constant 7 : i32
        %pad1_50 = arith.constant 1 : i32
        %outidx_50 = arith.constant 50 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_50, %smean_50, %sscale_50, %dilation_50, %padding_50, %ki0_50, %ki1_50, %ki2_50, %pad1_50, %out_v, %outidx_50) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 51: dilation=1 padding=4 kernel=(0,6,7) padding1=1
        %fidx_51 = arith.constant 51 : index
        %bidx_51 = arith.constant 302 : index
        %smidx_51 = arith.constant 722 : index
        %ssidx_51 = arith.constant 1142 : index
        %bias_51 = memref.load %in_v[%bidx_51] : memref<1511xi32>
        %smean_51 = memref.load %in_v[%smidx_51] : memref<1511xi32>
        %sscale_51 = memref.load %in_v[%ssidx_51] : memref<1511xi32>
        %dilation_51 = arith.constant 1 : i32
        %padding_51 = arith.constant 4 : i32
        %ki0_51 = arith.constant 0 : i32
        %ki1_51 = arith.constant 6 : i32
        %ki2_51 = arith.constant 7 : i32
        %pad1_51 = arith.constant 1 : i32
        %outidx_51 = arith.constant 51 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_51, %smean_51, %sscale_51, %dilation_51, %padding_51, %ki0_51, %ki1_51, %ki2_51, %pad1_51, %out_v, %outidx_51) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 52: dilation=1 padding=4 kernel=(0,6,8) padding1=0
        %fidx_52 = arith.constant 52 : index
        %bidx_52 = arith.constant 303 : index
        %smidx_52 = arith.constant 723 : index
        %ssidx_52 = arith.constant 1143 : index
        %bias_52 = memref.load %in_v[%bidx_52] : memref<1511xi32>
        %smean_52 = memref.load %in_v[%smidx_52] : memref<1511xi32>
        %sscale_52 = memref.load %in_v[%ssidx_52] : memref<1511xi32>
        %dilation_52 = arith.constant 1 : i32
        %padding_52 = arith.constant 4 : i32
        %ki0_52 = arith.constant 0 : i32
        %ki1_52 = arith.constant 6 : i32
        %ki2_52 = arith.constant 8 : i32
        %pad1_52 = arith.constant 0 : i32
        %outidx_52 = arith.constant 52 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_52, %smean_52, %sscale_52, %dilation_52, %padding_52, %ki0_52, %ki1_52, %ki2_52, %pad1_52, %out_v, %outidx_52) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 53: dilation=1 padding=4 kernel=(0,6,8) padding1=0
        %fidx_53 = arith.constant 53 : index
        %bidx_53 = arith.constant 304 : index
        %smidx_53 = arith.constant 724 : index
        %ssidx_53 = arith.constant 1144 : index
        %bias_53 = memref.load %in_v[%bidx_53] : memref<1511xi32>
        %smean_53 = memref.load %in_v[%smidx_53] : memref<1511xi32>
        %sscale_53 = memref.load %in_v[%ssidx_53] : memref<1511xi32>
        %dilation_53 = arith.constant 1 : i32
        %padding_53 = arith.constant 4 : i32
        %ki0_53 = arith.constant 0 : i32
        %ki1_53 = arith.constant 6 : i32
        %ki2_53 = arith.constant 8 : i32
        %pad1_53 = arith.constant 0 : i32
        %outidx_53 = arith.constant 53 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_53, %smean_53, %sscale_53, %dilation_53, %padding_53, %ki0_53, %ki1_53, %ki2_53, %pad1_53, %out_v, %outidx_53) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 54: dilation=1 padding=4 kernel=(0,7,8) padding1=1
        %fidx_54 = arith.constant 54 : index
        %bidx_54 = arith.constant 305 : index
        %smidx_54 = arith.constant 725 : index
        %ssidx_54 = arith.constant 1145 : index
        %bias_54 = memref.load %in_v[%bidx_54] : memref<1511xi32>
        %smean_54 = memref.load %in_v[%smidx_54] : memref<1511xi32>
        %sscale_54 = memref.load %in_v[%ssidx_54] : memref<1511xi32>
        %dilation_54 = arith.constant 1 : i32
        %padding_54 = arith.constant 4 : i32
        %ki0_54 = arith.constant 0 : i32
        %ki1_54 = arith.constant 7 : i32
        %ki2_54 = arith.constant 8 : i32
        %pad1_54 = arith.constant 1 : i32
        %outidx_54 = arith.constant 54 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_54, %smean_54, %sscale_54, %dilation_54, %padding_54, %ki0_54, %ki1_54, %ki2_54, %pad1_54, %out_v, %outidx_54) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 55: dilation=1 padding=4 kernel=(0,7,8) padding1=1
        %fidx_55 = arith.constant 55 : index
        %bidx_55 = arith.constant 306 : index
        %smidx_55 = arith.constant 726 : index
        %ssidx_55 = arith.constant 1146 : index
        %bias_55 = memref.load %in_v[%bidx_55] : memref<1511xi32>
        %smean_55 = memref.load %in_v[%smidx_55] : memref<1511xi32>
        %sscale_55 = memref.load %in_v[%ssidx_55] : memref<1511xi32>
        %dilation_55 = arith.constant 1 : i32
        %padding_55 = arith.constant 4 : i32
        %ki0_55 = arith.constant 0 : i32
        %ki1_55 = arith.constant 7 : i32
        %ki2_55 = arith.constant 8 : i32
        %pad1_55 = arith.constant 1 : i32
        %outidx_55 = arith.constant 55 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_55, %smean_55, %sscale_55, %dilation_55, %padding_55, %ki0_55, %ki1_55, %ki2_55, %pad1_55, %out_v, %outidx_55) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 56: dilation=1 padding=4 kernel=(1,2,3) padding1=0
        %fidx_56 = arith.constant 56 : index
        %bidx_56 = arith.constant 307 : index
        %smidx_56 = arith.constant 727 : index
        %ssidx_56 = arith.constant 1147 : index
        %bias_56 = memref.load %in_v[%bidx_56] : memref<1511xi32>
        %smean_56 = memref.load %in_v[%smidx_56] : memref<1511xi32>
        %sscale_56 = memref.load %in_v[%ssidx_56] : memref<1511xi32>
        %dilation_56 = arith.constant 1 : i32
        %padding_56 = arith.constant 4 : i32
        %ki0_56 = arith.constant 1 : i32
        %ki1_56 = arith.constant 2 : i32
        %ki2_56 = arith.constant 3 : i32
        %pad1_56 = arith.constant 0 : i32
        %outidx_56 = arith.constant 56 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_56, %smean_56, %sscale_56, %dilation_56, %padding_56, %ki0_56, %ki1_56, %ki2_56, %pad1_56, %out_v, %outidx_56) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 57: dilation=1 padding=4 kernel=(1,2,3) padding1=0
        %fidx_57 = arith.constant 57 : index
        %bidx_57 = arith.constant 308 : index
        %smidx_57 = arith.constant 728 : index
        %ssidx_57 = arith.constant 1148 : index
        %bias_57 = memref.load %in_v[%bidx_57] : memref<1511xi32>
        %smean_57 = memref.load %in_v[%smidx_57] : memref<1511xi32>
        %sscale_57 = memref.load %in_v[%ssidx_57] : memref<1511xi32>
        %dilation_57 = arith.constant 1 : i32
        %padding_57 = arith.constant 4 : i32
        %ki0_57 = arith.constant 1 : i32
        %ki1_57 = arith.constant 2 : i32
        %ki2_57 = arith.constant 3 : i32
        %pad1_57 = arith.constant 0 : i32
        %outidx_57 = arith.constant 57 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_57, %smean_57, %sscale_57, %dilation_57, %padding_57, %ki0_57, %ki1_57, %ki2_57, %pad1_57, %out_v, %outidx_57) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 58: dilation=1 padding=4 kernel=(1,2,4) padding1=1
        %fidx_58 = arith.constant 58 : index
        %bidx_58 = arith.constant 309 : index
        %smidx_58 = arith.constant 729 : index
        %ssidx_58 = arith.constant 1149 : index
        %bias_58 = memref.load %in_v[%bidx_58] : memref<1511xi32>
        %smean_58 = memref.load %in_v[%smidx_58] : memref<1511xi32>
        %sscale_58 = memref.load %in_v[%ssidx_58] : memref<1511xi32>
        %dilation_58 = arith.constant 1 : i32
        %padding_58 = arith.constant 4 : i32
        %ki0_58 = arith.constant 1 : i32
        %ki1_58 = arith.constant 2 : i32
        %ki2_58 = arith.constant 4 : i32
        %pad1_58 = arith.constant 1 : i32
        %outidx_58 = arith.constant 58 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_58, %smean_58, %sscale_58, %dilation_58, %padding_58, %ki0_58, %ki1_58, %ki2_58, %pad1_58, %out_v, %outidx_58) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 59: dilation=1 padding=4 kernel=(1,2,4) padding1=1
        %fidx_59 = arith.constant 59 : index
        %bidx_59 = arith.constant 310 : index
        %smidx_59 = arith.constant 730 : index
        %ssidx_59 = arith.constant 1150 : index
        %bias_59 = memref.load %in_v[%bidx_59] : memref<1511xi32>
        %smean_59 = memref.load %in_v[%smidx_59] : memref<1511xi32>
        %sscale_59 = memref.load %in_v[%ssidx_59] : memref<1511xi32>
        %dilation_59 = arith.constant 1 : i32
        %padding_59 = arith.constant 4 : i32
        %ki0_59 = arith.constant 1 : i32
        %ki1_59 = arith.constant 2 : i32
        %ki2_59 = arith.constant 4 : i32
        %pad1_59 = arith.constant 1 : i32
        %outidx_59 = arith.constant 59 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_59, %smean_59, %sscale_59, %dilation_59, %padding_59, %ki0_59, %ki1_59, %ki2_59, %pad1_59, %out_v, %outidx_59) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 60: dilation=1 padding=4 kernel=(1,2,5) padding1=0
        %fidx_60 = arith.constant 60 : index
        %bidx_60 = arith.constant 311 : index
        %smidx_60 = arith.constant 731 : index
        %ssidx_60 = arith.constant 1151 : index
        %bias_60 = memref.load %in_v[%bidx_60] : memref<1511xi32>
        %smean_60 = memref.load %in_v[%smidx_60] : memref<1511xi32>
        %sscale_60 = memref.load %in_v[%ssidx_60] : memref<1511xi32>
        %dilation_60 = arith.constant 1 : i32
        %padding_60 = arith.constant 4 : i32
        %ki0_60 = arith.constant 1 : i32
        %ki1_60 = arith.constant 2 : i32
        %ki2_60 = arith.constant 5 : i32
        %pad1_60 = arith.constant 0 : i32
        %outidx_60 = arith.constant 60 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_60, %smean_60, %sscale_60, %dilation_60, %padding_60, %ki0_60, %ki1_60, %ki2_60, %pad1_60, %out_v, %outidx_60) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 61: dilation=1 padding=4 kernel=(1,2,5) padding1=0
        %fidx_61 = arith.constant 61 : index
        %bidx_61 = arith.constant 312 : index
        %smidx_61 = arith.constant 732 : index
        %ssidx_61 = arith.constant 1152 : index
        %bias_61 = memref.load %in_v[%bidx_61] : memref<1511xi32>
        %smean_61 = memref.load %in_v[%smidx_61] : memref<1511xi32>
        %sscale_61 = memref.load %in_v[%ssidx_61] : memref<1511xi32>
        %dilation_61 = arith.constant 1 : i32
        %padding_61 = arith.constant 4 : i32
        %ki0_61 = arith.constant 1 : i32
        %ki1_61 = arith.constant 2 : i32
        %ki2_61 = arith.constant 5 : i32
        %pad1_61 = arith.constant 0 : i32
        %outidx_61 = arith.constant 61 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_61, %smean_61, %sscale_61, %dilation_61, %padding_61, %ki0_61, %ki1_61, %ki2_61, %pad1_61, %out_v, %outidx_61) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 62: dilation=1 padding=4 kernel=(1,2,6) padding1=1
        %fidx_62 = arith.constant 62 : index
        %bidx_62 = arith.constant 313 : index
        %smidx_62 = arith.constant 733 : index
        %ssidx_62 = arith.constant 1153 : index
        %bias_62 = memref.load %in_v[%bidx_62] : memref<1511xi32>
        %smean_62 = memref.load %in_v[%smidx_62] : memref<1511xi32>
        %sscale_62 = memref.load %in_v[%ssidx_62] : memref<1511xi32>
        %dilation_62 = arith.constant 1 : i32
        %padding_62 = arith.constant 4 : i32
        %ki0_62 = arith.constant 1 : i32
        %ki1_62 = arith.constant 2 : i32
        %ki2_62 = arith.constant 6 : i32
        %pad1_62 = arith.constant 1 : i32
        %outidx_62 = arith.constant 62 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_62, %smean_62, %sscale_62, %dilation_62, %padding_62, %ki0_62, %ki1_62, %ki2_62, %pad1_62, %out_v, %outidx_62) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 63: dilation=1 padding=4 kernel=(1,2,6) padding1=1
        %fidx_63 = arith.constant 63 : index
        %bidx_63 = arith.constant 314 : index
        %smidx_63 = arith.constant 734 : index
        %ssidx_63 = arith.constant 1154 : index
        %bias_63 = memref.load %in_v[%bidx_63] : memref<1511xi32>
        %smean_63 = memref.load %in_v[%smidx_63] : memref<1511xi32>
        %sscale_63 = memref.load %in_v[%ssidx_63] : memref<1511xi32>
        %dilation_63 = arith.constant 1 : i32
        %padding_63 = arith.constant 4 : i32
        %ki0_63 = arith.constant 1 : i32
        %ki1_63 = arith.constant 2 : i32
        %ki2_63 = arith.constant 6 : i32
        %pad1_63 = arith.constant 1 : i32
        %outidx_63 = arith.constant 63 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_63, %smean_63, %sscale_63, %dilation_63, %padding_63, %ki0_63, %ki1_63, %ki2_63, %pad1_63, %out_v, %outidx_63) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 64: dilation=1 padding=4 kernel=(1,2,7) padding1=0
        %fidx_64 = arith.constant 64 : index
        %bidx_64 = arith.constant 315 : index
        %smidx_64 = arith.constant 735 : index
        %ssidx_64 = arith.constant 1155 : index
        %bias_64 = memref.load %in_v[%bidx_64] : memref<1511xi32>
        %smean_64 = memref.load %in_v[%smidx_64] : memref<1511xi32>
        %sscale_64 = memref.load %in_v[%ssidx_64] : memref<1511xi32>
        %dilation_64 = arith.constant 1 : i32
        %padding_64 = arith.constant 4 : i32
        %ki0_64 = arith.constant 1 : i32
        %ki1_64 = arith.constant 2 : i32
        %ki2_64 = arith.constant 7 : i32
        %pad1_64 = arith.constant 0 : i32
        %outidx_64 = arith.constant 64 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_64, %smean_64, %sscale_64, %dilation_64, %padding_64, %ki0_64, %ki1_64, %ki2_64, %pad1_64, %out_v, %outidx_64) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 65: dilation=1 padding=4 kernel=(1,2,7) padding1=0
        %fidx_65 = arith.constant 65 : index
        %bidx_65 = arith.constant 316 : index
        %smidx_65 = arith.constant 736 : index
        %ssidx_65 = arith.constant 1156 : index
        %bias_65 = memref.load %in_v[%bidx_65] : memref<1511xi32>
        %smean_65 = memref.load %in_v[%smidx_65] : memref<1511xi32>
        %sscale_65 = memref.load %in_v[%ssidx_65] : memref<1511xi32>
        %dilation_65 = arith.constant 1 : i32
        %padding_65 = arith.constant 4 : i32
        %ki0_65 = arith.constant 1 : i32
        %ki1_65 = arith.constant 2 : i32
        %ki2_65 = arith.constant 7 : i32
        %pad1_65 = arith.constant 0 : i32
        %outidx_65 = arith.constant 65 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_65, %smean_65, %sscale_65, %dilation_65, %padding_65, %ki0_65, %ki1_65, %ki2_65, %pad1_65, %out_v, %outidx_65) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 66: dilation=1 padding=4 kernel=(1,2,8) padding1=1
        %fidx_66 = arith.constant 66 : index
        %bidx_66 = arith.constant 317 : index
        %smidx_66 = arith.constant 737 : index
        %ssidx_66 = arith.constant 1157 : index
        %bias_66 = memref.load %in_v[%bidx_66] : memref<1511xi32>
        %smean_66 = memref.load %in_v[%smidx_66] : memref<1511xi32>
        %sscale_66 = memref.load %in_v[%ssidx_66] : memref<1511xi32>
        %dilation_66 = arith.constant 1 : i32
        %padding_66 = arith.constant 4 : i32
        %ki0_66 = arith.constant 1 : i32
        %ki1_66 = arith.constant 2 : i32
        %ki2_66 = arith.constant 8 : i32
        %pad1_66 = arith.constant 1 : i32
        %outidx_66 = arith.constant 66 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_66, %smean_66, %sscale_66, %dilation_66, %padding_66, %ki0_66, %ki1_66, %ki2_66, %pad1_66, %out_v, %outidx_66) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 67: dilation=1 padding=4 kernel=(1,2,8) padding1=1
        %fidx_67 = arith.constant 67 : index
        %bidx_67 = arith.constant 318 : index
        %smidx_67 = arith.constant 738 : index
        %ssidx_67 = arith.constant 1158 : index
        %bias_67 = memref.load %in_v[%bidx_67] : memref<1511xi32>
        %smean_67 = memref.load %in_v[%smidx_67] : memref<1511xi32>
        %sscale_67 = memref.load %in_v[%ssidx_67] : memref<1511xi32>
        %dilation_67 = arith.constant 1 : i32
        %padding_67 = arith.constant 4 : i32
        %ki0_67 = arith.constant 1 : i32
        %ki1_67 = arith.constant 2 : i32
        %ki2_67 = arith.constant 8 : i32
        %pad1_67 = arith.constant 1 : i32
        %outidx_67 = arith.constant 67 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_67, %smean_67, %sscale_67, %dilation_67, %padding_67, %ki0_67, %ki1_67, %ki2_67, %pad1_67, %out_v, %outidx_67) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 68: dilation=1 padding=4 kernel=(1,3,4) padding1=0
        %fidx_68 = arith.constant 68 : index
        %bidx_68 = arith.constant 319 : index
        %smidx_68 = arith.constant 739 : index
        %ssidx_68 = arith.constant 1159 : index
        %bias_68 = memref.load %in_v[%bidx_68] : memref<1511xi32>
        %smean_68 = memref.load %in_v[%smidx_68] : memref<1511xi32>
        %sscale_68 = memref.load %in_v[%ssidx_68] : memref<1511xi32>
        %dilation_68 = arith.constant 1 : i32
        %padding_68 = arith.constant 4 : i32
        %ki0_68 = arith.constant 1 : i32
        %ki1_68 = arith.constant 3 : i32
        %ki2_68 = arith.constant 4 : i32
        %pad1_68 = arith.constant 0 : i32
        %outidx_68 = arith.constant 68 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_68, %smean_68, %sscale_68, %dilation_68, %padding_68, %ki0_68, %ki1_68, %ki2_68, %pad1_68, %out_v, %outidx_68) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 69: dilation=1 padding=4 kernel=(1,3,4) padding1=0
        %fidx_69 = arith.constant 69 : index
        %bidx_69 = arith.constant 320 : index
        %smidx_69 = arith.constant 740 : index
        %ssidx_69 = arith.constant 1160 : index
        %bias_69 = memref.load %in_v[%bidx_69] : memref<1511xi32>
        %smean_69 = memref.load %in_v[%smidx_69] : memref<1511xi32>
        %sscale_69 = memref.load %in_v[%ssidx_69] : memref<1511xi32>
        %dilation_69 = arith.constant 1 : i32
        %padding_69 = arith.constant 4 : i32
        %ki0_69 = arith.constant 1 : i32
        %ki1_69 = arith.constant 3 : i32
        %ki2_69 = arith.constant 4 : i32
        %pad1_69 = arith.constant 0 : i32
        %outidx_69 = arith.constant 69 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_69, %smean_69, %sscale_69, %dilation_69, %padding_69, %ki0_69, %ki1_69, %ki2_69, %pad1_69, %out_v, %outidx_69) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 70: dilation=1 padding=4 kernel=(1,3,5) padding1=1
        %fidx_70 = arith.constant 70 : index
        %bidx_70 = arith.constant 321 : index
        %smidx_70 = arith.constant 741 : index
        %ssidx_70 = arith.constant 1161 : index
        %bias_70 = memref.load %in_v[%bidx_70] : memref<1511xi32>
        %smean_70 = memref.load %in_v[%smidx_70] : memref<1511xi32>
        %sscale_70 = memref.load %in_v[%ssidx_70] : memref<1511xi32>
        %dilation_70 = arith.constant 1 : i32
        %padding_70 = arith.constant 4 : i32
        %ki0_70 = arith.constant 1 : i32
        %ki1_70 = arith.constant 3 : i32
        %ki2_70 = arith.constant 5 : i32
        %pad1_70 = arith.constant 1 : i32
        %outidx_70 = arith.constant 70 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_70, %smean_70, %sscale_70, %dilation_70, %padding_70, %ki0_70, %ki1_70, %ki2_70, %pad1_70, %out_v, %outidx_70) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 71: dilation=1 padding=4 kernel=(1,3,5) padding1=1
        %fidx_71 = arith.constant 71 : index
        %bidx_71 = arith.constant 322 : index
        %smidx_71 = arith.constant 742 : index
        %ssidx_71 = arith.constant 1162 : index
        %bias_71 = memref.load %in_v[%bidx_71] : memref<1511xi32>
        %smean_71 = memref.load %in_v[%smidx_71] : memref<1511xi32>
        %sscale_71 = memref.load %in_v[%ssidx_71] : memref<1511xi32>
        %dilation_71 = arith.constant 1 : i32
        %padding_71 = arith.constant 4 : i32
        %ki0_71 = arith.constant 1 : i32
        %ki1_71 = arith.constant 3 : i32
        %ki2_71 = arith.constant 5 : i32
        %pad1_71 = arith.constant 1 : i32
        %outidx_71 = arith.constant 71 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_71, %smean_71, %sscale_71, %dilation_71, %padding_71, %ki0_71, %ki1_71, %ki2_71, %pad1_71, %out_v, %outidx_71) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 72: dilation=1 padding=4 kernel=(1,3,6) padding1=0
        %fidx_72 = arith.constant 72 : index
        %bidx_72 = arith.constant 323 : index
        %smidx_72 = arith.constant 743 : index
        %ssidx_72 = arith.constant 1163 : index
        %bias_72 = memref.load %in_v[%bidx_72] : memref<1511xi32>
        %smean_72 = memref.load %in_v[%smidx_72] : memref<1511xi32>
        %sscale_72 = memref.load %in_v[%ssidx_72] : memref<1511xi32>
        %dilation_72 = arith.constant 1 : i32
        %padding_72 = arith.constant 4 : i32
        %ki0_72 = arith.constant 1 : i32
        %ki1_72 = arith.constant 3 : i32
        %ki2_72 = arith.constant 6 : i32
        %pad1_72 = arith.constant 0 : i32
        %outidx_72 = arith.constant 72 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_72, %smean_72, %sscale_72, %dilation_72, %padding_72, %ki0_72, %ki1_72, %ki2_72, %pad1_72, %out_v, %outidx_72) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 73: dilation=1 padding=4 kernel=(1,3,6) padding1=0
        %fidx_73 = arith.constant 73 : index
        %bidx_73 = arith.constant 324 : index
        %smidx_73 = arith.constant 744 : index
        %ssidx_73 = arith.constant 1164 : index
        %bias_73 = memref.load %in_v[%bidx_73] : memref<1511xi32>
        %smean_73 = memref.load %in_v[%smidx_73] : memref<1511xi32>
        %sscale_73 = memref.load %in_v[%ssidx_73] : memref<1511xi32>
        %dilation_73 = arith.constant 1 : i32
        %padding_73 = arith.constant 4 : i32
        %ki0_73 = arith.constant 1 : i32
        %ki1_73 = arith.constant 3 : i32
        %ki2_73 = arith.constant 6 : i32
        %pad1_73 = arith.constant 0 : i32
        %outidx_73 = arith.constant 73 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_73, %smean_73, %sscale_73, %dilation_73, %padding_73, %ki0_73, %ki1_73, %ki2_73, %pad1_73, %out_v, %outidx_73) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 74: dilation=1 padding=4 kernel=(1,3,7) padding1=1
        %fidx_74 = arith.constant 74 : index
        %bidx_74 = arith.constant 325 : index
        %smidx_74 = arith.constant 745 : index
        %ssidx_74 = arith.constant 1165 : index
        %bias_74 = memref.load %in_v[%bidx_74] : memref<1511xi32>
        %smean_74 = memref.load %in_v[%smidx_74] : memref<1511xi32>
        %sscale_74 = memref.load %in_v[%ssidx_74] : memref<1511xi32>
        %dilation_74 = arith.constant 1 : i32
        %padding_74 = arith.constant 4 : i32
        %ki0_74 = arith.constant 1 : i32
        %ki1_74 = arith.constant 3 : i32
        %ki2_74 = arith.constant 7 : i32
        %pad1_74 = arith.constant 1 : i32
        %outidx_74 = arith.constant 74 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_74, %smean_74, %sscale_74, %dilation_74, %padding_74, %ki0_74, %ki1_74, %ki2_74, %pad1_74, %out_v, %outidx_74) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 75: dilation=1 padding=4 kernel=(1,3,7) padding1=1
        %fidx_75 = arith.constant 75 : index
        %bidx_75 = arith.constant 326 : index
        %smidx_75 = arith.constant 746 : index
        %ssidx_75 = arith.constant 1166 : index
        %bias_75 = memref.load %in_v[%bidx_75] : memref<1511xi32>
        %smean_75 = memref.load %in_v[%smidx_75] : memref<1511xi32>
        %sscale_75 = memref.load %in_v[%ssidx_75] : memref<1511xi32>
        %dilation_75 = arith.constant 1 : i32
        %padding_75 = arith.constant 4 : i32
        %ki0_75 = arith.constant 1 : i32
        %ki1_75 = arith.constant 3 : i32
        %ki2_75 = arith.constant 7 : i32
        %pad1_75 = arith.constant 1 : i32
        %outidx_75 = arith.constant 75 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_75, %smean_75, %sscale_75, %dilation_75, %padding_75, %ki0_75, %ki1_75, %ki2_75, %pad1_75, %out_v, %outidx_75) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 76: dilation=1 padding=4 kernel=(1,3,8) padding1=0
        %fidx_76 = arith.constant 76 : index
        %bidx_76 = arith.constant 327 : index
        %smidx_76 = arith.constant 747 : index
        %ssidx_76 = arith.constant 1167 : index
        %bias_76 = memref.load %in_v[%bidx_76] : memref<1511xi32>
        %smean_76 = memref.load %in_v[%smidx_76] : memref<1511xi32>
        %sscale_76 = memref.load %in_v[%ssidx_76] : memref<1511xi32>
        %dilation_76 = arith.constant 1 : i32
        %padding_76 = arith.constant 4 : i32
        %ki0_76 = arith.constant 1 : i32
        %ki1_76 = arith.constant 3 : i32
        %ki2_76 = arith.constant 8 : i32
        %pad1_76 = arith.constant 0 : i32
        %outidx_76 = arith.constant 76 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_76, %smean_76, %sscale_76, %dilation_76, %padding_76, %ki0_76, %ki1_76, %ki2_76, %pad1_76, %out_v, %outidx_76) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 77: dilation=1 padding=4 kernel=(1,3,8) padding1=0
        %fidx_77 = arith.constant 77 : index
        %bidx_77 = arith.constant 328 : index
        %smidx_77 = arith.constant 748 : index
        %ssidx_77 = arith.constant 1168 : index
        %bias_77 = memref.load %in_v[%bidx_77] : memref<1511xi32>
        %smean_77 = memref.load %in_v[%smidx_77] : memref<1511xi32>
        %sscale_77 = memref.load %in_v[%ssidx_77] : memref<1511xi32>
        %dilation_77 = arith.constant 1 : i32
        %padding_77 = arith.constant 4 : i32
        %ki0_77 = arith.constant 1 : i32
        %ki1_77 = arith.constant 3 : i32
        %ki2_77 = arith.constant 8 : i32
        %pad1_77 = arith.constant 0 : i32
        %outidx_77 = arith.constant 77 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_77, %smean_77, %sscale_77, %dilation_77, %padding_77, %ki0_77, %ki1_77, %ki2_77, %pad1_77, %out_v, %outidx_77) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 78: dilation=1 padding=4 kernel=(1,4,5) padding1=1
        %fidx_78 = arith.constant 78 : index
        %bidx_78 = arith.constant 329 : index
        %smidx_78 = arith.constant 749 : index
        %ssidx_78 = arith.constant 1169 : index
        %bias_78 = memref.load %in_v[%bidx_78] : memref<1511xi32>
        %smean_78 = memref.load %in_v[%smidx_78] : memref<1511xi32>
        %sscale_78 = memref.load %in_v[%ssidx_78] : memref<1511xi32>
        %dilation_78 = arith.constant 1 : i32
        %padding_78 = arith.constant 4 : i32
        %ki0_78 = arith.constant 1 : i32
        %ki1_78 = arith.constant 4 : i32
        %ki2_78 = arith.constant 5 : i32
        %pad1_78 = arith.constant 1 : i32
        %outidx_78 = arith.constant 78 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_78, %smean_78, %sscale_78, %dilation_78, %padding_78, %ki0_78, %ki1_78, %ki2_78, %pad1_78, %out_v, %outidx_78) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 79: dilation=1 padding=4 kernel=(1,4,5) padding1=1
        %fidx_79 = arith.constant 79 : index
        %bidx_79 = arith.constant 330 : index
        %smidx_79 = arith.constant 750 : index
        %ssidx_79 = arith.constant 1170 : index
        %bias_79 = memref.load %in_v[%bidx_79] : memref<1511xi32>
        %smean_79 = memref.load %in_v[%smidx_79] : memref<1511xi32>
        %sscale_79 = memref.load %in_v[%ssidx_79] : memref<1511xi32>
        %dilation_79 = arith.constant 1 : i32
        %padding_79 = arith.constant 4 : i32
        %ki0_79 = arith.constant 1 : i32
        %ki1_79 = arith.constant 4 : i32
        %ki2_79 = arith.constant 5 : i32
        %pad1_79 = arith.constant 1 : i32
        %outidx_79 = arith.constant 79 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_79, %smean_79, %sscale_79, %dilation_79, %padding_79, %ki0_79, %ki1_79, %ki2_79, %pad1_79, %out_v, %outidx_79) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 80: dilation=1 padding=4 kernel=(1,4,6) padding1=0
        %fidx_80 = arith.constant 80 : index
        %bidx_80 = arith.constant 331 : index
        %smidx_80 = arith.constant 751 : index
        %ssidx_80 = arith.constant 1171 : index
        %bias_80 = memref.load %in_v[%bidx_80] : memref<1511xi32>
        %smean_80 = memref.load %in_v[%smidx_80] : memref<1511xi32>
        %sscale_80 = memref.load %in_v[%ssidx_80] : memref<1511xi32>
        %dilation_80 = arith.constant 1 : i32
        %padding_80 = arith.constant 4 : i32
        %ki0_80 = arith.constant 1 : i32
        %ki1_80 = arith.constant 4 : i32
        %ki2_80 = arith.constant 6 : i32
        %pad1_80 = arith.constant 0 : i32
        %outidx_80 = arith.constant 80 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_80, %smean_80, %sscale_80, %dilation_80, %padding_80, %ki0_80, %ki1_80, %ki2_80, %pad1_80, %out_v, %outidx_80) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 81: dilation=1 padding=4 kernel=(1,4,6) padding1=0
        %fidx_81 = arith.constant 81 : index
        %bidx_81 = arith.constant 332 : index
        %smidx_81 = arith.constant 752 : index
        %ssidx_81 = arith.constant 1172 : index
        %bias_81 = memref.load %in_v[%bidx_81] : memref<1511xi32>
        %smean_81 = memref.load %in_v[%smidx_81] : memref<1511xi32>
        %sscale_81 = memref.load %in_v[%ssidx_81] : memref<1511xi32>
        %dilation_81 = arith.constant 1 : i32
        %padding_81 = arith.constant 4 : i32
        %ki0_81 = arith.constant 1 : i32
        %ki1_81 = arith.constant 4 : i32
        %ki2_81 = arith.constant 6 : i32
        %pad1_81 = arith.constant 0 : i32
        %outidx_81 = arith.constant 81 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_81, %smean_81, %sscale_81, %dilation_81, %padding_81, %ki0_81, %ki1_81, %ki2_81, %pad1_81, %out_v, %outidx_81) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 82: dilation=1 padding=4 kernel=(1,4,7) padding1=1
        %fidx_82 = arith.constant 82 : index
        %bidx_82 = arith.constant 333 : index
        %smidx_82 = arith.constant 753 : index
        %ssidx_82 = arith.constant 1173 : index
        %bias_82 = memref.load %in_v[%bidx_82] : memref<1511xi32>
        %smean_82 = memref.load %in_v[%smidx_82] : memref<1511xi32>
        %sscale_82 = memref.load %in_v[%ssidx_82] : memref<1511xi32>
        %dilation_82 = arith.constant 1 : i32
        %padding_82 = arith.constant 4 : i32
        %ki0_82 = arith.constant 1 : i32
        %ki1_82 = arith.constant 4 : i32
        %ki2_82 = arith.constant 7 : i32
        %pad1_82 = arith.constant 1 : i32
        %outidx_82 = arith.constant 82 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_82, %smean_82, %sscale_82, %dilation_82, %padding_82, %ki0_82, %ki1_82, %ki2_82, %pad1_82, %out_v, %outidx_82) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 83: dilation=1 padding=4 kernel=(1,4,7) padding1=1
        %fidx_83 = arith.constant 83 : index
        %bidx_83 = arith.constant 334 : index
        %smidx_83 = arith.constant 754 : index
        %ssidx_83 = arith.constant 1174 : index
        %bias_83 = memref.load %in_v[%bidx_83] : memref<1511xi32>
        %smean_83 = memref.load %in_v[%smidx_83] : memref<1511xi32>
        %sscale_83 = memref.load %in_v[%ssidx_83] : memref<1511xi32>
        %dilation_83 = arith.constant 1 : i32
        %padding_83 = arith.constant 4 : i32
        %ki0_83 = arith.constant 1 : i32
        %ki1_83 = arith.constant 4 : i32
        %ki2_83 = arith.constant 7 : i32
        %pad1_83 = arith.constant 1 : i32
        %outidx_83 = arith.constant 83 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_83, %smean_83, %sscale_83, %dilation_83, %padding_83, %ki0_83, %ki1_83, %ki2_83, %pad1_83, %out_v, %outidx_83) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 84: dilation=1 padding=4 kernel=(1,4,8) padding1=0
        %fidx_84 = arith.constant 84 : index
        %bidx_84 = arith.constant 335 : index
        %smidx_84 = arith.constant 755 : index
        %ssidx_84 = arith.constant 1175 : index
        %bias_84 = memref.load %in_v[%bidx_84] : memref<1511xi32>
        %smean_84 = memref.load %in_v[%smidx_84] : memref<1511xi32>
        %sscale_84 = memref.load %in_v[%ssidx_84] : memref<1511xi32>
        %dilation_84 = arith.constant 1 : i32
        %padding_84 = arith.constant 4 : i32
        %ki0_84 = arith.constant 1 : i32
        %ki1_84 = arith.constant 4 : i32
        %ki2_84 = arith.constant 8 : i32
        %pad1_84 = arith.constant 0 : i32
        %outidx_84 = arith.constant 84 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_84, %smean_84, %sscale_84, %dilation_84, %padding_84, %ki0_84, %ki1_84, %ki2_84, %pad1_84, %out_v, %outidx_84) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 85: dilation=1 padding=4 kernel=(1,4,8) padding1=0
        %fidx_85 = arith.constant 85 : index
        %bidx_85 = arith.constant 336 : index
        %smidx_85 = arith.constant 756 : index
        %ssidx_85 = arith.constant 1176 : index
        %bias_85 = memref.load %in_v[%bidx_85] : memref<1511xi32>
        %smean_85 = memref.load %in_v[%smidx_85] : memref<1511xi32>
        %sscale_85 = memref.load %in_v[%ssidx_85] : memref<1511xi32>
        %dilation_85 = arith.constant 1 : i32
        %padding_85 = arith.constant 4 : i32
        %ki0_85 = arith.constant 1 : i32
        %ki1_85 = arith.constant 4 : i32
        %ki2_85 = arith.constant 8 : i32
        %pad1_85 = arith.constant 0 : i32
        %outidx_85 = arith.constant 85 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_85, %smean_85, %sscale_85, %dilation_85, %padding_85, %ki0_85, %ki1_85, %ki2_85, %pad1_85, %out_v, %outidx_85) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 86: dilation=1 padding=4 kernel=(1,5,6) padding1=1
        %fidx_86 = arith.constant 86 : index
        %bidx_86 = arith.constant 337 : index
        %smidx_86 = arith.constant 757 : index
        %ssidx_86 = arith.constant 1177 : index
        %bias_86 = memref.load %in_v[%bidx_86] : memref<1511xi32>
        %smean_86 = memref.load %in_v[%smidx_86] : memref<1511xi32>
        %sscale_86 = memref.load %in_v[%ssidx_86] : memref<1511xi32>
        %dilation_86 = arith.constant 1 : i32
        %padding_86 = arith.constant 4 : i32
        %ki0_86 = arith.constant 1 : i32
        %ki1_86 = arith.constant 5 : i32
        %ki2_86 = arith.constant 6 : i32
        %pad1_86 = arith.constant 1 : i32
        %outidx_86 = arith.constant 86 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_86, %smean_86, %sscale_86, %dilation_86, %padding_86, %ki0_86, %ki1_86, %ki2_86, %pad1_86, %out_v, %outidx_86) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 87: dilation=1 padding=4 kernel=(1,5,6) padding1=1
        %fidx_87 = arith.constant 87 : index
        %bidx_87 = arith.constant 338 : index
        %smidx_87 = arith.constant 758 : index
        %ssidx_87 = arith.constant 1178 : index
        %bias_87 = memref.load %in_v[%bidx_87] : memref<1511xi32>
        %smean_87 = memref.load %in_v[%smidx_87] : memref<1511xi32>
        %sscale_87 = memref.load %in_v[%ssidx_87] : memref<1511xi32>
        %dilation_87 = arith.constant 1 : i32
        %padding_87 = arith.constant 4 : i32
        %ki0_87 = arith.constant 1 : i32
        %ki1_87 = arith.constant 5 : i32
        %ki2_87 = arith.constant 6 : i32
        %pad1_87 = arith.constant 1 : i32
        %outidx_87 = arith.constant 87 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_87, %smean_87, %sscale_87, %dilation_87, %padding_87, %ki0_87, %ki1_87, %ki2_87, %pad1_87, %out_v, %outidx_87) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 88: dilation=1 padding=4 kernel=(1,5,7) padding1=0
        %fidx_88 = arith.constant 88 : index
        %bidx_88 = arith.constant 339 : index
        %smidx_88 = arith.constant 759 : index
        %ssidx_88 = arith.constant 1179 : index
        %bias_88 = memref.load %in_v[%bidx_88] : memref<1511xi32>
        %smean_88 = memref.load %in_v[%smidx_88] : memref<1511xi32>
        %sscale_88 = memref.load %in_v[%ssidx_88] : memref<1511xi32>
        %dilation_88 = arith.constant 1 : i32
        %padding_88 = arith.constant 4 : i32
        %ki0_88 = arith.constant 1 : i32
        %ki1_88 = arith.constant 5 : i32
        %ki2_88 = arith.constant 7 : i32
        %pad1_88 = arith.constant 0 : i32
        %outidx_88 = arith.constant 88 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_88, %smean_88, %sscale_88, %dilation_88, %padding_88, %ki0_88, %ki1_88, %ki2_88, %pad1_88, %out_v, %outidx_88) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 89: dilation=1 padding=4 kernel=(1,5,7) padding1=0
        %fidx_89 = arith.constant 89 : index
        %bidx_89 = arith.constant 340 : index
        %smidx_89 = arith.constant 760 : index
        %ssidx_89 = arith.constant 1180 : index
        %bias_89 = memref.load %in_v[%bidx_89] : memref<1511xi32>
        %smean_89 = memref.load %in_v[%smidx_89] : memref<1511xi32>
        %sscale_89 = memref.load %in_v[%ssidx_89] : memref<1511xi32>
        %dilation_89 = arith.constant 1 : i32
        %padding_89 = arith.constant 4 : i32
        %ki0_89 = arith.constant 1 : i32
        %ki1_89 = arith.constant 5 : i32
        %ki2_89 = arith.constant 7 : i32
        %pad1_89 = arith.constant 0 : i32
        %outidx_89 = arith.constant 89 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_89, %smean_89, %sscale_89, %dilation_89, %padding_89, %ki0_89, %ki1_89, %ki2_89, %pad1_89, %out_v, %outidx_89) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 90: dilation=1 padding=4 kernel=(1,5,8) padding1=1
        %fidx_90 = arith.constant 90 : index
        %bidx_90 = arith.constant 341 : index
        %smidx_90 = arith.constant 761 : index
        %ssidx_90 = arith.constant 1181 : index
        %bias_90 = memref.load %in_v[%bidx_90] : memref<1511xi32>
        %smean_90 = memref.load %in_v[%smidx_90] : memref<1511xi32>
        %sscale_90 = memref.load %in_v[%ssidx_90] : memref<1511xi32>
        %dilation_90 = arith.constant 1 : i32
        %padding_90 = arith.constant 4 : i32
        %ki0_90 = arith.constant 1 : i32
        %ki1_90 = arith.constant 5 : i32
        %ki2_90 = arith.constant 8 : i32
        %pad1_90 = arith.constant 1 : i32
        %outidx_90 = arith.constant 90 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_90, %smean_90, %sscale_90, %dilation_90, %padding_90, %ki0_90, %ki1_90, %ki2_90, %pad1_90, %out_v, %outidx_90) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 91: dilation=1 padding=4 kernel=(1,5,8) padding1=1
        %fidx_91 = arith.constant 91 : index
        %bidx_91 = arith.constant 342 : index
        %smidx_91 = arith.constant 762 : index
        %ssidx_91 = arith.constant 1182 : index
        %bias_91 = memref.load %in_v[%bidx_91] : memref<1511xi32>
        %smean_91 = memref.load %in_v[%smidx_91] : memref<1511xi32>
        %sscale_91 = memref.load %in_v[%ssidx_91] : memref<1511xi32>
        %dilation_91 = arith.constant 1 : i32
        %padding_91 = arith.constant 4 : i32
        %ki0_91 = arith.constant 1 : i32
        %ki1_91 = arith.constant 5 : i32
        %ki2_91 = arith.constant 8 : i32
        %pad1_91 = arith.constant 1 : i32
        %outidx_91 = arith.constant 91 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_91, %smean_91, %sscale_91, %dilation_91, %padding_91, %ki0_91, %ki1_91, %ki2_91, %pad1_91, %out_v, %outidx_91) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 92: dilation=1 padding=4 kernel=(1,6,7) padding1=0
        %fidx_92 = arith.constant 92 : index
        %bidx_92 = arith.constant 343 : index
        %smidx_92 = arith.constant 763 : index
        %ssidx_92 = arith.constant 1183 : index
        %bias_92 = memref.load %in_v[%bidx_92] : memref<1511xi32>
        %smean_92 = memref.load %in_v[%smidx_92] : memref<1511xi32>
        %sscale_92 = memref.load %in_v[%ssidx_92] : memref<1511xi32>
        %dilation_92 = arith.constant 1 : i32
        %padding_92 = arith.constant 4 : i32
        %ki0_92 = arith.constant 1 : i32
        %ki1_92 = arith.constant 6 : i32
        %ki2_92 = arith.constant 7 : i32
        %pad1_92 = arith.constant 0 : i32
        %outidx_92 = arith.constant 92 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_92, %smean_92, %sscale_92, %dilation_92, %padding_92, %ki0_92, %ki1_92, %ki2_92, %pad1_92, %out_v, %outidx_92) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 93: dilation=1 padding=4 kernel=(1,6,7) padding1=0
        %fidx_93 = arith.constant 93 : index
        %bidx_93 = arith.constant 344 : index
        %smidx_93 = arith.constant 764 : index
        %ssidx_93 = arith.constant 1184 : index
        %bias_93 = memref.load %in_v[%bidx_93] : memref<1511xi32>
        %smean_93 = memref.load %in_v[%smidx_93] : memref<1511xi32>
        %sscale_93 = memref.load %in_v[%ssidx_93] : memref<1511xi32>
        %dilation_93 = arith.constant 1 : i32
        %padding_93 = arith.constant 4 : i32
        %ki0_93 = arith.constant 1 : i32
        %ki1_93 = arith.constant 6 : i32
        %ki2_93 = arith.constant 7 : i32
        %pad1_93 = arith.constant 0 : i32
        %outidx_93 = arith.constant 93 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_93, %smean_93, %sscale_93, %dilation_93, %padding_93, %ki0_93, %ki1_93, %ki2_93, %pad1_93, %out_v, %outidx_93) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 94: dilation=1 padding=4 kernel=(1,6,8) padding1=1
        %fidx_94 = arith.constant 94 : index
        %bidx_94 = arith.constant 345 : index
        %smidx_94 = arith.constant 765 : index
        %ssidx_94 = arith.constant 1185 : index
        %bias_94 = memref.load %in_v[%bidx_94] : memref<1511xi32>
        %smean_94 = memref.load %in_v[%smidx_94] : memref<1511xi32>
        %sscale_94 = memref.load %in_v[%ssidx_94] : memref<1511xi32>
        %dilation_94 = arith.constant 1 : i32
        %padding_94 = arith.constant 4 : i32
        %ki0_94 = arith.constant 1 : i32
        %ki1_94 = arith.constant 6 : i32
        %ki2_94 = arith.constant 8 : i32
        %pad1_94 = arith.constant 1 : i32
        %outidx_94 = arith.constant 94 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_94, %smean_94, %sscale_94, %dilation_94, %padding_94, %ki0_94, %ki1_94, %ki2_94, %pad1_94, %out_v, %outidx_94) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 95: dilation=1 padding=4 kernel=(1,6,8) padding1=1
        %fidx_95 = arith.constant 95 : index
        %bidx_95 = arith.constant 346 : index
        %smidx_95 = arith.constant 766 : index
        %ssidx_95 = arith.constant 1186 : index
        %bias_95 = memref.load %in_v[%bidx_95] : memref<1511xi32>
        %smean_95 = memref.load %in_v[%smidx_95] : memref<1511xi32>
        %sscale_95 = memref.load %in_v[%ssidx_95] : memref<1511xi32>
        %dilation_95 = arith.constant 1 : i32
        %padding_95 = arith.constant 4 : i32
        %ki0_95 = arith.constant 1 : i32
        %ki1_95 = arith.constant 6 : i32
        %ki2_95 = arith.constant 8 : i32
        %pad1_95 = arith.constant 1 : i32
        %outidx_95 = arith.constant 95 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_95, %smean_95, %sscale_95, %dilation_95, %padding_95, %ki0_95, %ki1_95, %ki2_95, %pad1_95, %out_v, %outidx_95) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 96: dilation=1 padding=4 kernel=(1,7,8) padding1=0
        %fidx_96 = arith.constant 96 : index
        %bidx_96 = arith.constant 347 : index
        %smidx_96 = arith.constant 767 : index
        %ssidx_96 = arith.constant 1187 : index
        %bias_96 = memref.load %in_v[%bidx_96] : memref<1511xi32>
        %smean_96 = memref.load %in_v[%smidx_96] : memref<1511xi32>
        %sscale_96 = memref.load %in_v[%ssidx_96] : memref<1511xi32>
        %dilation_96 = arith.constant 1 : i32
        %padding_96 = arith.constant 4 : i32
        %ki0_96 = arith.constant 1 : i32
        %ki1_96 = arith.constant 7 : i32
        %ki2_96 = arith.constant 8 : i32
        %pad1_96 = arith.constant 0 : i32
        %outidx_96 = arith.constant 96 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_96, %smean_96, %sscale_96, %dilation_96, %padding_96, %ki0_96, %ki1_96, %ki2_96, %pad1_96, %out_v, %outidx_96) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 97: dilation=1 padding=4 kernel=(1,7,8) padding1=0
        %fidx_97 = arith.constant 97 : index
        %bidx_97 = arith.constant 348 : index
        %smidx_97 = arith.constant 768 : index
        %ssidx_97 = arith.constant 1188 : index
        %bias_97 = memref.load %in_v[%bidx_97] : memref<1511xi32>
        %smean_97 = memref.load %in_v[%smidx_97] : memref<1511xi32>
        %sscale_97 = memref.load %in_v[%ssidx_97] : memref<1511xi32>
        %dilation_97 = arith.constant 1 : i32
        %padding_97 = arith.constant 4 : i32
        %ki0_97 = arith.constant 1 : i32
        %ki1_97 = arith.constant 7 : i32
        %ki2_97 = arith.constant 8 : i32
        %pad1_97 = arith.constant 0 : i32
        %outidx_97 = arith.constant 97 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_97, %smean_97, %sscale_97, %dilation_97, %padding_97, %ki0_97, %ki1_97, %ki2_97, %pad1_97, %out_v, %outidx_97) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 98: dilation=1 padding=4 kernel=(2,3,4) padding1=1
        %fidx_98 = arith.constant 98 : index
        %bidx_98 = arith.constant 349 : index
        %smidx_98 = arith.constant 769 : index
        %ssidx_98 = arith.constant 1189 : index
        %bias_98 = memref.load %in_v[%bidx_98] : memref<1511xi32>
        %smean_98 = memref.load %in_v[%smidx_98] : memref<1511xi32>
        %sscale_98 = memref.load %in_v[%ssidx_98] : memref<1511xi32>
        %dilation_98 = arith.constant 1 : i32
        %padding_98 = arith.constant 4 : i32
        %ki0_98 = arith.constant 2 : i32
        %ki1_98 = arith.constant 3 : i32
        %ki2_98 = arith.constant 4 : i32
        %pad1_98 = arith.constant 1 : i32
        %outidx_98 = arith.constant 98 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_98, %smean_98, %sscale_98, %dilation_98, %padding_98, %ki0_98, %ki1_98, %ki2_98, %pad1_98, %out_v, %outidx_98) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 99: dilation=1 padding=4 kernel=(2,3,4) padding1=1
        %fidx_99 = arith.constant 99 : index
        %bidx_99 = arith.constant 350 : index
        %smidx_99 = arith.constant 770 : index
        %ssidx_99 = arith.constant 1190 : index
        %bias_99 = memref.load %in_v[%bidx_99] : memref<1511xi32>
        %smean_99 = memref.load %in_v[%smidx_99] : memref<1511xi32>
        %sscale_99 = memref.load %in_v[%ssidx_99] : memref<1511xi32>
        %dilation_99 = arith.constant 1 : i32
        %padding_99 = arith.constant 4 : i32
        %ki0_99 = arith.constant 2 : i32
        %ki1_99 = arith.constant 3 : i32
        %ki2_99 = arith.constant 4 : i32
        %pad1_99 = arith.constant 1 : i32
        %outidx_99 = arith.constant 99 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_99, %smean_99, %sscale_99, %dilation_99, %padding_99, %ki0_99, %ki1_99, %ki2_99, %pad1_99, %out_v, %outidx_99) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 100: dilation=1 padding=4 kernel=(2,3,5) padding1=0
        %fidx_100 = arith.constant 100 : index
        %bidx_100 = arith.constant 351 : index
        %smidx_100 = arith.constant 771 : index
        %ssidx_100 = arith.constant 1191 : index
        %bias_100 = memref.load %in_v[%bidx_100] : memref<1511xi32>
        %smean_100 = memref.load %in_v[%smidx_100] : memref<1511xi32>
        %sscale_100 = memref.load %in_v[%ssidx_100] : memref<1511xi32>
        %dilation_100 = arith.constant 1 : i32
        %padding_100 = arith.constant 4 : i32
        %ki0_100 = arith.constant 2 : i32
        %ki1_100 = arith.constant 3 : i32
        %ki2_100 = arith.constant 5 : i32
        %pad1_100 = arith.constant 0 : i32
        %outidx_100 = arith.constant 100 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_100, %smean_100, %sscale_100, %dilation_100, %padding_100, %ki0_100, %ki1_100, %ki2_100, %pad1_100, %out_v, %outidx_100) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 101: dilation=1 padding=4 kernel=(2,3,5) padding1=0
        %fidx_101 = arith.constant 101 : index
        %bidx_101 = arith.constant 352 : index
        %smidx_101 = arith.constant 772 : index
        %ssidx_101 = arith.constant 1192 : index
        %bias_101 = memref.load %in_v[%bidx_101] : memref<1511xi32>
        %smean_101 = memref.load %in_v[%smidx_101] : memref<1511xi32>
        %sscale_101 = memref.load %in_v[%ssidx_101] : memref<1511xi32>
        %dilation_101 = arith.constant 1 : i32
        %padding_101 = arith.constant 4 : i32
        %ki0_101 = arith.constant 2 : i32
        %ki1_101 = arith.constant 3 : i32
        %ki2_101 = arith.constant 5 : i32
        %pad1_101 = arith.constant 0 : i32
        %outidx_101 = arith.constant 101 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_101, %smean_101, %sscale_101, %dilation_101, %padding_101, %ki0_101, %ki1_101, %ki2_101, %pad1_101, %out_v, %outidx_101) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 102: dilation=1 padding=4 kernel=(2,3,6) padding1=1
        %fidx_102 = arith.constant 102 : index
        %bidx_102 = arith.constant 353 : index
        %smidx_102 = arith.constant 773 : index
        %ssidx_102 = arith.constant 1193 : index
        %bias_102 = memref.load %in_v[%bidx_102] : memref<1511xi32>
        %smean_102 = memref.load %in_v[%smidx_102] : memref<1511xi32>
        %sscale_102 = memref.load %in_v[%ssidx_102] : memref<1511xi32>
        %dilation_102 = arith.constant 1 : i32
        %padding_102 = arith.constant 4 : i32
        %ki0_102 = arith.constant 2 : i32
        %ki1_102 = arith.constant 3 : i32
        %ki2_102 = arith.constant 6 : i32
        %pad1_102 = arith.constant 1 : i32
        %outidx_102 = arith.constant 102 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_102, %smean_102, %sscale_102, %dilation_102, %padding_102, %ki0_102, %ki1_102, %ki2_102, %pad1_102, %out_v, %outidx_102) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 103: dilation=1 padding=4 kernel=(2,3,6) padding1=1
        %fidx_103 = arith.constant 103 : index
        %bidx_103 = arith.constant 354 : index
        %smidx_103 = arith.constant 774 : index
        %ssidx_103 = arith.constant 1194 : index
        %bias_103 = memref.load %in_v[%bidx_103] : memref<1511xi32>
        %smean_103 = memref.load %in_v[%smidx_103] : memref<1511xi32>
        %sscale_103 = memref.load %in_v[%ssidx_103] : memref<1511xi32>
        %dilation_103 = arith.constant 1 : i32
        %padding_103 = arith.constant 4 : i32
        %ki0_103 = arith.constant 2 : i32
        %ki1_103 = arith.constant 3 : i32
        %ki2_103 = arith.constant 6 : i32
        %pad1_103 = arith.constant 1 : i32
        %outidx_103 = arith.constant 103 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_103, %smean_103, %sscale_103, %dilation_103, %padding_103, %ki0_103, %ki1_103, %ki2_103, %pad1_103, %out_v, %outidx_103) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 104: dilation=1 padding=4 kernel=(2,3,7) padding1=0
        %fidx_104 = arith.constant 104 : index
        %bidx_104 = arith.constant 355 : index
        %smidx_104 = arith.constant 775 : index
        %ssidx_104 = arith.constant 1195 : index
        %bias_104 = memref.load %in_v[%bidx_104] : memref<1511xi32>
        %smean_104 = memref.load %in_v[%smidx_104] : memref<1511xi32>
        %sscale_104 = memref.load %in_v[%ssidx_104] : memref<1511xi32>
        %dilation_104 = arith.constant 1 : i32
        %padding_104 = arith.constant 4 : i32
        %ki0_104 = arith.constant 2 : i32
        %ki1_104 = arith.constant 3 : i32
        %ki2_104 = arith.constant 7 : i32
        %pad1_104 = arith.constant 0 : i32
        %outidx_104 = arith.constant 104 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_104, %smean_104, %sscale_104, %dilation_104, %padding_104, %ki0_104, %ki1_104, %ki2_104, %pad1_104, %out_v, %outidx_104) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 105: dilation=1 padding=4 kernel=(2,3,7) padding1=0
        %fidx_105 = arith.constant 105 : index
        %bidx_105 = arith.constant 356 : index
        %smidx_105 = arith.constant 776 : index
        %ssidx_105 = arith.constant 1196 : index
        %bias_105 = memref.load %in_v[%bidx_105] : memref<1511xi32>
        %smean_105 = memref.load %in_v[%smidx_105] : memref<1511xi32>
        %sscale_105 = memref.load %in_v[%ssidx_105] : memref<1511xi32>
        %dilation_105 = arith.constant 1 : i32
        %padding_105 = arith.constant 4 : i32
        %ki0_105 = arith.constant 2 : i32
        %ki1_105 = arith.constant 3 : i32
        %ki2_105 = arith.constant 7 : i32
        %pad1_105 = arith.constant 0 : i32
        %outidx_105 = arith.constant 105 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_105, %smean_105, %sscale_105, %dilation_105, %padding_105, %ki0_105, %ki1_105, %ki2_105, %pad1_105, %out_v, %outidx_105) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 106: dilation=1 padding=4 kernel=(2,3,8) padding1=1
        %fidx_106 = arith.constant 106 : index
        %bidx_106 = arith.constant 357 : index
        %smidx_106 = arith.constant 777 : index
        %ssidx_106 = arith.constant 1197 : index
        %bias_106 = memref.load %in_v[%bidx_106] : memref<1511xi32>
        %smean_106 = memref.load %in_v[%smidx_106] : memref<1511xi32>
        %sscale_106 = memref.load %in_v[%ssidx_106] : memref<1511xi32>
        %dilation_106 = arith.constant 1 : i32
        %padding_106 = arith.constant 4 : i32
        %ki0_106 = arith.constant 2 : i32
        %ki1_106 = arith.constant 3 : i32
        %ki2_106 = arith.constant 8 : i32
        %pad1_106 = arith.constant 1 : i32
        %outidx_106 = arith.constant 106 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_106, %smean_106, %sscale_106, %dilation_106, %padding_106, %ki0_106, %ki1_106, %ki2_106, %pad1_106, %out_v, %outidx_106) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 107: dilation=1 padding=4 kernel=(2,3,8) padding1=1
        %fidx_107 = arith.constant 107 : index
        %bidx_107 = arith.constant 358 : index
        %smidx_107 = arith.constant 778 : index
        %ssidx_107 = arith.constant 1198 : index
        %bias_107 = memref.load %in_v[%bidx_107] : memref<1511xi32>
        %smean_107 = memref.load %in_v[%smidx_107] : memref<1511xi32>
        %sscale_107 = memref.load %in_v[%ssidx_107] : memref<1511xi32>
        %dilation_107 = arith.constant 1 : i32
        %padding_107 = arith.constant 4 : i32
        %ki0_107 = arith.constant 2 : i32
        %ki1_107 = arith.constant 3 : i32
        %ki2_107 = arith.constant 8 : i32
        %pad1_107 = arith.constant 1 : i32
        %outidx_107 = arith.constant 107 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_107, %smean_107, %sscale_107, %dilation_107, %padding_107, %ki0_107, %ki1_107, %ki2_107, %pad1_107, %out_v, %outidx_107) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 108: dilation=1 padding=4 kernel=(2,4,5) padding1=0
        %fidx_108 = arith.constant 108 : index
        %bidx_108 = arith.constant 359 : index
        %smidx_108 = arith.constant 779 : index
        %ssidx_108 = arith.constant 1199 : index
        %bias_108 = memref.load %in_v[%bidx_108] : memref<1511xi32>
        %smean_108 = memref.load %in_v[%smidx_108] : memref<1511xi32>
        %sscale_108 = memref.load %in_v[%ssidx_108] : memref<1511xi32>
        %dilation_108 = arith.constant 1 : i32
        %padding_108 = arith.constant 4 : i32
        %ki0_108 = arith.constant 2 : i32
        %ki1_108 = arith.constant 4 : i32
        %ki2_108 = arith.constant 5 : i32
        %pad1_108 = arith.constant 0 : i32
        %outidx_108 = arith.constant 108 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_108, %smean_108, %sscale_108, %dilation_108, %padding_108, %ki0_108, %ki1_108, %ki2_108, %pad1_108, %out_v, %outidx_108) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 109: dilation=1 padding=4 kernel=(2,4,5) padding1=0
        %fidx_109 = arith.constant 109 : index
        %bidx_109 = arith.constant 360 : index
        %smidx_109 = arith.constant 780 : index
        %ssidx_109 = arith.constant 1200 : index
        %bias_109 = memref.load %in_v[%bidx_109] : memref<1511xi32>
        %smean_109 = memref.load %in_v[%smidx_109] : memref<1511xi32>
        %sscale_109 = memref.load %in_v[%ssidx_109] : memref<1511xi32>
        %dilation_109 = arith.constant 1 : i32
        %padding_109 = arith.constant 4 : i32
        %ki0_109 = arith.constant 2 : i32
        %ki1_109 = arith.constant 4 : i32
        %ki2_109 = arith.constant 5 : i32
        %pad1_109 = arith.constant 0 : i32
        %outidx_109 = arith.constant 109 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_109, %smean_109, %sscale_109, %dilation_109, %padding_109, %ki0_109, %ki1_109, %ki2_109, %pad1_109, %out_v, %outidx_109) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 110: dilation=1 padding=4 kernel=(2,4,6) padding1=1
        %fidx_110 = arith.constant 110 : index
        %bidx_110 = arith.constant 361 : index
        %smidx_110 = arith.constant 781 : index
        %ssidx_110 = arith.constant 1201 : index
        %bias_110 = memref.load %in_v[%bidx_110] : memref<1511xi32>
        %smean_110 = memref.load %in_v[%smidx_110] : memref<1511xi32>
        %sscale_110 = memref.load %in_v[%ssidx_110] : memref<1511xi32>
        %dilation_110 = arith.constant 1 : i32
        %padding_110 = arith.constant 4 : i32
        %ki0_110 = arith.constant 2 : i32
        %ki1_110 = arith.constant 4 : i32
        %ki2_110 = arith.constant 6 : i32
        %pad1_110 = arith.constant 1 : i32
        %outidx_110 = arith.constant 110 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_110, %smean_110, %sscale_110, %dilation_110, %padding_110, %ki0_110, %ki1_110, %ki2_110, %pad1_110, %out_v, %outidx_110) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 111: dilation=1 padding=4 kernel=(2,4,6) padding1=1
        %fidx_111 = arith.constant 111 : index
        %bidx_111 = arith.constant 362 : index
        %smidx_111 = arith.constant 782 : index
        %ssidx_111 = arith.constant 1202 : index
        %bias_111 = memref.load %in_v[%bidx_111] : memref<1511xi32>
        %smean_111 = memref.load %in_v[%smidx_111] : memref<1511xi32>
        %sscale_111 = memref.load %in_v[%ssidx_111] : memref<1511xi32>
        %dilation_111 = arith.constant 1 : i32
        %padding_111 = arith.constant 4 : i32
        %ki0_111 = arith.constant 2 : i32
        %ki1_111 = arith.constant 4 : i32
        %ki2_111 = arith.constant 6 : i32
        %pad1_111 = arith.constant 1 : i32
        %outidx_111 = arith.constant 111 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_111, %smean_111, %sscale_111, %dilation_111, %padding_111, %ki0_111, %ki1_111, %ki2_111, %pad1_111, %out_v, %outidx_111) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 112: dilation=1 padding=4 kernel=(2,4,7) padding1=0
        %fidx_112 = arith.constant 112 : index
        %bidx_112 = arith.constant 363 : index
        %smidx_112 = arith.constant 783 : index
        %ssidx_112 = arith.constant 1203 : index
        %bias_112 = memref.load %in_v[%bidx_112] : memref<1511xi32>
        %smean_112 = memref.load %in_v[%smidx_112] : memref<1511xi32>
        %sscale_112 = memref.load %in_v[%ssidx_112] : memref<1511xi32>
        %dilation_112 = arith.constant 1 : i32
        %padding_112 = arith.constant 4 : i32
        %ki0_112 = arith.constant 2 : i32
        %ki1_112 = arith.constant 4 : i32
        %ki2_112 = arith.constant 7 : i32
        %pad1_112 = arith.constant 0 : i32
        %outidx_112 = arith.constant 112 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_112, %smean_112, %sscale_112, %dilation_112, %padding_112, %ki0_112, %ki1_112, %ki2_112, %pad1_112, %out_v, %outidx_112) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 113: dilation=1 padding=4 kernel=(2,4,7) padding1=0
        %fidx_113 = arith.constant 113 : index
        %bidx_113 = arith.constant 364 : index
        %smidx_113 = arith.constant 784 : index
        %ssidx_113 = arith.constant 1204 : index
        %bias_113 = memref.load %in_v[%bidx_113] : memref<1511xi32>
        %smean_113 = memref.load %in_v[%smidx_113] : memref<1511xi32>
        %sscale_113 = memref.load %in_v[%ssidx_113] : memref<1511xi32>
        %dilation_113 = arith.constant 1 : i32
        %padding_113 = arith.constant 4 : i32
        %ki0_113 = arith.constant 2 : i32
        %ki1_113 = arith.constant 4 : i32
        %ki2_113 = arith.constant 7 : i32
        %pad1_113 = arith.constant 0 : i32
        %outidx_113 = arith.constant 113 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_113, %smean_113, %sscale_113, %dilation_113, %padding_113, %ki0_113, %ki1_113, %ki2_113, %pad1_113, %out_v, %outidx_113) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 114: dilation=1 padding=4 kernel=(2,4,8) padding1=1
        %fidx_114 = arith.constant 114 : index
        %bidx_114 = arith.constant 365 : index
        %smidx_114 = arith.constant 785 : index
        %ssidx_114 = arith.constant 1205 : index
        %bias_114 = memref.load %in_v[%bidx_114] : memref<1511xi32>
        %smean_114 = memref.load %in_v[%smidx_114] : memref<1511xi32>
        %sscale_114 = memref.load %in_v[%ssidx_114] : memref<1511xi32>
        %dilation_114 = arith.constant 1 : i32
        %padding_114 = arith.constant 4 : i32
        %ki0_114 = arith.constant 2 : i32
        %ki1_114 = arith.constant 4 : i32
        %ki2_114 = arith.constant 8 : i32
        %pad1_114 = arith.constant 1 : i32
        %outidx_114 = arith.constant 114 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_114, %smean_114, %sscale_114, %dilation_114, %padding_114, %ki0_114, %ki1_114, %ki2_114, %pad1_114, %out_v, %outidx_114) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 115: dilation=1 padding=4 kernel=(2,4,8) padding1=1
        %fidx_115 = arith.constant 115 : index
        %bidx_115 = arith.constant 366 : index
        %smidx_115 = arith.constant 786 : index
        %ssidx_115 = arith.constant 1206 : index
        %bias_115 = memref.load %in_v[%bidx_115] : memref<1511xi32>
        %smean_115 = memref.load %in_v[%smidx_115] : memref<1511xi32>
        %sscale_115 = memref.load %in_v[%ssidx_115] : memref<1511xi32>
        %dilation_115 = arith.constant 1 : i32
        %padding_115 = arith.constant 4 : i32
        %ki0_115 = arith.constant 2 : i32
        %ki1_115 = arith.constant 4 : i32
        %ki2_115 = arith.constant 8 : i32
        %pad1_115 = arith.constant 1 : i32
        %outidx_115 = arith.constant 115 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_115, %smean_115, %sscale_115, %dilation_115, %padding_115, %ki0_115, %ki1_115, %ki2_115, %pad1_115, %out_v, %outidx_115) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 116: dilation=1 padding=4 kernel=(2,5,6) padding1=0
        %fidx_116 = arith.constant 116 : index
        %bidx_116 = arith.constant 367 : index
        %smidx_116 = arith.constant 787 : index
        %ssidx_116 = arith.constant 1207 : index
        %bias_116 = memref.load %in_v[%bidx_116] : memref<1511xi32>
        %smean_116 = memref.load %in_v[%smidx_116] : memref<1511xi32>
        %sscale_116 = memref.load %in_v[%ssidx_116] : memref<1511xi32>
        %dilation_116 = arith.constant 1 : i32
        %padding_116 = arith.constant 4 : i32
        %ki0_116 = arith.constant 2 : i32
        %ki1_116 = arith.constant 5 : i32
        %ki2_116 = arith.constant 6 : i32
        %pad1_116 = arith.constant 0 : i32
        %outidx_116 = arith.constant 116 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_116, %smean_116, %sscale_116, %dilation_116, %padding_116, %ki0_116, %ki1_116, %ki2_116, %pad1_116, %out_v, %outidx_116) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 117: dilation=1 padding=4 kernel=(2,5,6) padding1=0
        %fidx_117 = arith.constant 117 : index
        %bidx_117 = arith.constant 368 : index
        %smidx_117 = arith.constant 788 : index
        %ssidx_117 = arith.constant 1208 : index
        %bias_117 = memref.load %in_v[%bidx_117] : memref<1511xi32>
        %smean_117 = memref.load %in_v[%smidx_117] : memref<1511xi32>
        %sscale_117 = memref.load %in_v[%ssidx_117] : memref<1511xi32>
        %dilation_117 = arith.constant 1 : i32
        %padding_117 = arith.constant 4 : i32
        %ki0_117 = arith.constant 2 : i32
        %ki1_117 = arith.constant 5 : i32
        %ki2_117 = arith.constant 6 : i32
        %pad1_117 = arith.constant 0 : i32
        %outidx_117 = arith.constant 117 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_117, %smean_117, %sscale_117, %dilation_117, %padding_117, %ki0_117, %ki1_117, %ki2_117, %pad1_117, %out_v, %outidx_117) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 118: dilation=1 padding=4 kernel=(2,5,7) padding1=1
        %fidx_118 = arith.constant 118 : index
        %bidx_118 = arith.constant 369 : index
        %smidx_118 = arith.constant 789 : index
        %ssidx_118 = arith.constant 1209 : index
        %bias_118 = memref.load %in_v[%bidx_118] : memref<1511xi32>
        %smean_118 = memref.load %in_v[%smidx_118] : memref<1511xi32>
        %sscale_118 = memref.load %in_v[%ssidx_118] : memref<1511xi32>
        %dilation_118 = arith.constant 1 : i32
        %padding_118 = arith.constant 4 : i32
        %ki0_118 = arith.constant 2 : i32
        %ki1_118 = arith.constant 5 : i32
        %ki2_118 = arith.constant 7 : i32
        %pad1_118 = arith.constant 1 : i32
        %outidx_118 = arith.constant 118 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_118, %smean_118, %sscale_118, %dilation_118, %padding_118, %ki0_118, %ki1_118, %ki2_118, %pad1_118, %out_v, %outidx_118) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 119: dilation=1 padding=4 kernel=(2,5,7) padding1=1
        %fidx_119 = arith.constant 119 : index
        %bidx_119 = arith.constant 370 : index
        %smidx_119 = arith.constant 790 : index
        %ssidx_119 = arith.constant 1210 : index
        %bias_119 = memref.load %in_v[%bidx_119] : memref<1511xi32>
        %smean_119 = memref.load %in_v[%smidx_119] : memref<1511xi32>
        %sscale_119 = memref.load %in_v[%ssidx_119] : memref<1511xi32>
        %dilation_119 = arith.constant 1 : i32
        %padding_119 = arith.constant 4 : i32
        %ki0_119 = arith.constant 2 : i32
        %ki1_119 = arith.constant 5 : i32
        %ki2_119 = arith.constant 7 : i32
        %pad1_119 = arith.constant 1 : i32
        %outidx_119 = arith.constant 119 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_119, %smean_119, %sscale_119, %dilation_119, %padding_119, %ki0_119, %ki1_119, %ki2_119, %pad1_119, %out_v, %outidx_119) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 120: dilation=1 padding=4 kernel=(2,5,8) padding1=0
        %fidx_120 = arith.constant 120 : index
        %bidx_120 = arith.constant 371 : index
        %smidx_120 = arith.constant 791 : index
        %ssidx_120 = arith.constant 1211 : index
        %bias_120 = memref.load %in_v[%bidx_120] : memref<1511xi32>
        %smean_120 = memref.load %in_v[%smidx_120] : memref<1511xi32>
        %sscale_120 = memref.load %in_v[%ssidx_120] : memref<1511xi32>
        %dilation_120 = arith.constant 1 : i32
        %padding_120 = arith.constant 4 : i32
        %ki0_120 = arith.constant 2 : i32
        %ki1_120 = arith.constant 5 : i32
        %ki2_120 = arith.constant 8 : i32
        %pad1_120 = arith.constant 0 : i32
        %outidx_120 = arith.constant 120 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_120, %smean_120, %sscale_120, %dilation_120, %padding_120, %ki0_120, %ki1_120, %ki2_120, %pad1_120, %out_v, %outidx_120) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 121: dilation=1 padding=4 kernel=(2,5,8) padding1=0
        %fidx_121 = arith.constant 121 : index
        %bidx_121 = arith.constant 372 : index
        %smidx_121 = arith.constant 792 : index
        %ssidx_121 = arith.constant 1212 : index
        %bias_121 = memref.load %in_v[%bidx_121] : memref<1511xi32>
        %smean_121 = memref.load %in_v[%smidx_121] : memref<1511xi32>
        %sscale_121 = memref.load %in_v[%ssidx_121] : memref<1511xi32>
        %dilation_121 = arith.constant 1 : i32
        %padding_121 = arith.constant 4 : i32
        %ki0_121 = arith.constant 2 : i32
        %ki1_121 = arith.constant 5 : i32
        %ki2_121 = arith.constant 8 : i32
        %pad1_121 = arith.constant 0 : i32
        %outidx_121 = arith.constant 121 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_121, %smean_121, %sscale_121, %dilation_121, %padding_121, %ki0_121, %ki1_121, %ki2_121, %pad1_121, %out_v, %outidx_121) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 122: dilation=1 padding=4 kernel=(2,6,7) padding1=1
        %fidx_122 = arith.constant 122 : index
        %bidx_122 = arith.constant 373 : index
        %smidx_122 = arith.constant 793 : index
        %ssidx_122 = arith.constant 1213 : index
        %bias_122 = memref.load %in_v[%bidx_122] : memref<1511xi32>
        %smean_122 = memref.load %in_v[%smidx_122] : memref<1511xi32>
        %sscale_122 = memref.load %in_v[%ssidx_122] : memref<1511xi32>
        %dilation_122 = arith.constant 1 : i32
        %padding_122 = arith.constant 4 : i32
        %ki0_122 = arith.constant 2 : i32
        %ki1_122 = arith.constant 6 : i32
        %ki2_122 = arith.constant 7 : i32
        %pad1_122 = arith.constant 1 : i32
        %outidx_122 = arith.constant 122 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_122, %smean_122, %sscale_122, %dilation_122, %padding_122, %ki0_122, %ki1_122, %ki2_122, %pad1_122, %out_v, %outidx_122) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 123: dilation=1 padding=4 kernel=(2,6,7) padding1=1
        %fidx_123 = arith.constant 123 : index
        %bidx_123 = arith.constant 374 : index
        %smidx_123 = arith.constant 794 : index
        %ssidx_123 = arith.constant 1214 : index
        %bias_123 = memref.load %in_v[%bidx_123] : memref<1511xi32>
        %smean_123 = memref.load %in_v[%smidx_123] : memref<1511xi32>
        %sscale_123 = memref.load %in_v[%ssidx_123] : memref<1511xi32>
        %dilation_123 = arith.constant 1 : i32
        %padding_123 = arith.constant 4 : i32
        %ki0_123 = arith.constant 2 : i32
        %ki1_123 = arith.constant 6 : i32
        %ki2_123 = arith.constant 7 : i32
        %pad1_123 = arith.constant 1 : i32
        %outidx_123 = arith.constant 123 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_123, %smean_123, %sscale_123, %dilation_123, %padding_123, %ki0_123, %ki1_123, %ki2_123, %pad1_123, %out_v, %outidx_123) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 124: dilation=1 padding=4 kernel=(2,6,8) padding1=0
        %fidx_124 = arith.constant 124 : index
        %bidx_124 = arith.constant 375 : index
        %smidx_124 = arith.constant 795 : index
        %ssidx_124 = arith.constant 1215 : index
        %bias_124 = memref.load %in_v[%bidx_124] : memref<1511xi32>
        %smean_124 = memref.load %in_v[%smidx_124] : memref<1511xi32>
        %sscale_124 = memref.load %in_v[%ssidx_124] : memref<1511xi32>
        %dilation_124 = arith.constant 1 : i32
        %padding_124 = arith.constant 4 : i32
        %ki0_124 = arith.constant 2 : i32
        %ki1_124 = arith.constant 6 : i32
        %ki2_124 = arith.constant 8 : i32
        %pad1_124 = arith.constant 0 : i32
        %outidx_124 = arith.constant 124 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_124, %smean_124, %sscale_124, %dilation_124, %padding_124, %ki0_124, %ki1_124, %ki2_124, %pad1_124, %out_v, %outidx_124) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 125: dilation=1 padding=4 kernel=(2,6,8) padding1=0
        %fidx_125 = arith.constant 125 : index
        %bidx_125 = arith.constant 376 : index
        %smidx_125 = arith.constant 796 : index
        %ssidx_125 = arith.constant 1216 : index
        %bias_125 = memref.load %in_v[%bidx_125] : memref<1511xi32>
        %smean_125 = memref.load %in_v[%smidx_125] : memref<1511xi32>
        %sscale_125 = memref.load %in_v[%ssidx_125] : memref<1511xi32>
        %dilation_125 = arith.constant 1 : i32
        %padding_125 = arith.constant 4 : i32
        %ki0_125 = arith.constant 2 : i32
        %ki1_125 = arith.constant 6 : i32
        %ki2_125 = arith.constant 8 : i32
        %pad1_125 = arith.constant 0 : i32
        %outidx_125 = arith.constant 125 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_125, %smean_125, %sscale_125, %dilation_125, %padding_125, %ki0_125, %ki1_125, %ki2_125, %pad1_125, %out_v, %outidx_125) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 126: dilation=1 padding=4 kernel=(2,7,8) padding1=1
        %fidx_126 = arith.constant 126 : index
        %bidx_126 = arith.constant 377 : index
        %smidx_126 = arith.constant 797 : index
        %ssidx_126 = arith.constant 1217 : index
        %bias_126 = memref.load %in_v[%bidx_126] : memref<1511xi32>
        %smean_126 = memref.load %in_v[%smidx_126] : memref<1511xi32>
        %sscale_126 = memref.load %in_v[%ssidx_126] : memref<1511xi32>
        %dilation_126 = arith.constant 1 : i32
        %padding_126 = arith.constant 4 : i32
        %ki0_126 = arith.constant 2 : i32
        %ki1_126 = arith.constant 7 : i32
        %ki2_126 = arith.constant 8 : i32
        %pad1_126 = arith.constant 1 : i32
        %outidx_126 = arith.constant 126 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_126, %smean_126, %sscale_126, %dilation_126, %padding_126, %ki0_126, %ki1_126, %ki2_126, %pad1_126, %out_v, %outidx_126) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 127: dilation=1 padding=4 kernel=(2,7,8) padding1=1
        %fidx_127 = arith.constant 127 : index
        %bidx_127 = arith.constant 378 : index
        %smidx_127 = arith.constant 798 : index
        %ssidx_127 = arith.constant 1218 : index
        %bias_127 = memref.load %in_v[%bidx_127] : memref<1511xi32>
        %smean_127 = memref.load %in_v[%smidx_127] : memref<1511xi32>
        %sscale_127 = memref.load %in_v[%ssidx_127] : memref<1511xi32>
        %dilation_127 = arith.constant 1 : i32
        %padding_127 = arith.constant 4 : i32
        %ki0_127 = arith.constant 2 : i32
        %ki1_127 = arith.constant 7 : i32
        %ki2_127 = arith.constant 8 : i32
        %pad1_127 = arith.constant 1 : i32
        %outidx_127 = arith.constant 127 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_127, %smean_127, %sscale_127, %dilation_127, %padding_127, %ki0_127, %ki1_127, %ki2_127, %pad1_127, %out_v, %outidx_127) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 128: dilation=1 padding=4 kernel=(3,4,5) padding1=0
        %fidx_128 = arith.constant 128 : index
        %bidx_128 = arith.constant 379 : index
        %smidx_128 = arith.constant 799 : index
        %ssidx_128 = arith.constant 1219 : index
        %bias_128 = memref.load %in_v[%bidx_128] : memref<1511xi32>
        %smean_128 = memref.load %in_v[%smidx_128] : memref<1511xi32>
        %sscale_128 = memref.load %in_v[%ssidx_128] : memref<1511xi32>
        %dilation_128 = arith.constant 1 : i32
        %padding_128 = arith.constant 4 : i32
        %ki0_128 = arith.constant 3 : i32
        %ki1_128 = arith.constant 4 : i32
        %ki2_128 = arith.constant 5 : i32
        %pad1_128 = arith.constant 0 : i32
        %outidx_128 = arith.constant 128 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_128, %smean_128, %sscale_128, %dilation_128, %padding_128, %ki0_128, %ki1_128, %ki2_128, %pad1_128, %out_v, %outidx_128) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 129: dilation=1 padding=4 kernel=(3,4,5) padding1=0
        %fidx_129 = arith.constant 129 : index
        %bidx_129 = arith.constant 380 : index
        %smidx_129 = arith.constant 800 : index
        %ssidx_129 = arith.constant 1220 : index
        %bias_129 = memref.load %in_v[%bidx_129] : memref<1511xi32>
        %smean_129 = memref.load %in_v[%smidx_129] : memref<1511xi32>
        %sscale_129 = memref.load %in_v[%ssidx_129] : memref<1511xi32>
        %dilation_129 = arith.constant 1 : i32
        %padding_129 = arith.constant 4 : i32
        %ki0_129 = arith.constant 3 : i32
        %ki1_129 = arith.constant 4 : i32
        %ki2_129 = arith.constant 5 : i32
        %pad1_129 = arith.constant 0 : i32
        %outidx_129 = arith.constant 129 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_129, %smean_129, %sscale_129, %dilation_129, %padding_129, %ki0_129, %ki1_129, %ki2_129, %pad1_129, %out_v, %outidx_129) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 130: dilation=1 padding=4 kernel=(3,4,6) padding1=1
        %fidx_130 = arith.constant 130 : index
        %bidx_130 = arith.constant 381 : index
        %smidx_130 = arith.constant 801 : index
        %ssidx_130 = arith.constant 1221 : index
        %bias_130 = memref.load %in_v[%bidx_130] : memref<1511xi32>
        %smean_130 = memref.load %in_v[%smidx_130] : memref<1511xi32>
        %sscale_130 = memref.load %in_v[%ssidx_130] : memref<1511xi32>
        %dilation_130 = arith.constant 1 : i32
        %padding_130 = arith.constant 4 : i32
        %ki0_130 = arith.constant 3 : i32
        %ki1_130 = arith.constant 4 : i32
        %ki2_130 = arith.constant 6 : i32
        %pad1_130 = arith.constant 1 : i32
        %outidx_130 = arith.constant 130 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_130, %smean_130, %sscale_130, %dilation_130, %padding_130, %ki0_130, %ki1_130, %ki2_130, %pad1_130, %out_v, %outidx_130) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 131: dilation=1 padding=4 kernel=(3,4,6) padding1=1
        %fidx_131 = arith.constant 131 : index
        %bidx_131 = arith.constant 382 : index
        %smidx_131 = arith.constant 802 : index
        %ssidx_131 = arith.constant 1222 : index
        %bias_131 = memref.load %in_v[%bidx_131] : memref<1511xi32>
        %smean_131 = memref.load %in_v[%smidx_131] : memref<1511xi32>
        %sscale_131 = memref.load %in_v[%ssidx_131] : memref<1511xi32>
        %dilation_131 = arith.constant 1 : i32
        %padding_131 = arith.constant 4 : i32
        %ki0_131 = arith.constant 3 : i32
        %ki1_131 = arith.constant 4 : i32
        %ki2_131 = arith.constant 6 : i32
        %pad1_131 = arith.constant 1 : i32
        %outidx_131 = arith.constant 131 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_131, %smean_131, %sscale_131, %dilation_131, %padding_131, %ki0_131, %ki1_131, %ki2_131, %pad1_131, %out_v, %outidx_131) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 132: dilation=1 padding=4 kernel=(3,4,7) padding1=0
        %fidx_132 = arith.constant 132 : index
        %bidx_132 = arith.constant 383 : index
        %smidx_132 = arith.constant 803 : index
        %ssidx_132 = arith.constant 1223 : index
        %bias_132 = memref.load %in_v[%bidx_132] : memref<1511xi32>
        %smean_132 = memref.load %in_v[%smidx_132] : memref<1511xi32>
        %sscale_132 = memref.load %in_v[%ssidx_132] : memref<1511xi32>
        %dilation_132 = arith.constant 1 : i32
        %padding_132 = arith.constant 4 : i32
        %ki0_132 = arith.constant 3 : i32
        %ki1_132 = arith.constant 4 : i32
        %ki2_132 = arith.constant 7 : i32
        %pad1_132 = arith.constant 0 : i32
        %outidx_132 = arith.constant 132 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_132, %smean_132, %sscale_132, %dilation_132, %padding_132, %ki0_132, %ki1_132, %ki2_132, %pad1_132, %out_v, %outidx_132) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 133: dilation=1 padding=4 kernel=(3,4,7) padding1=0
        %fidx_133 = arith.constant 133 : index
        %bidx_133 = arith.constant 384 : index
        %smidx_133 = arith.constant 804 : index
        %ssidx_133 = arith.constant 1224 : index
        %bias_133 = memref.load %in_v[%bidx_133] : memref<1511xi32>
        %smean_133 = memref.load %in_v[%smidx_133] : memref<1511xi32>
        %sscale_133 = memref.load %in_v[%ssidx_133] : memref<1511xi32>
        %dilation_133 = arith.constant 1 : i32
        %padding_133 = arith.constant 4 : i32
        %ki0_133 = arith.constant 3 : i32
        %ki1_133 = arith.constant 4 : i32
        %ki2_133 = arith.constant 7 : i32
        %pad1_133 = arith.constant 0 : i32
        %outidx_133 = arith.constant 133 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_133, %smean_133, %sscale_133, %dilation_133, %padding_133, %ki0_133, %ki1_133, %ki2_133, %pad1_133, %out_v, %outidx_133) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 134: dilation=1 padding=4 kernel=(3,4,8) padding1=1
        %fidx_134 = arith.constant 134 : index
        %bidx_134 = arith.constant 385 : index
        %smidx_134 = arith.constant 805 : index
        %ssidx_134 = arith.constant 1225 : index
        %bias_134 = memref.load %in_v[%bidx_134] : memref<1511xi32>
        %smean_134 = memref.load %in_v[%smidx_134] : memref<1511xi32>
        %sscale_134 = memref.load %in_v[%ssidx_134] : memref<1511xi32>
        %dilation_134 = arith.constant 1 : i32
        %padding_134 = arith.constant 4 : i32
        %ki0_134 = arith.constant 3 : i32
        %ki1_134 = arith.constant 4 : i32
        %ki2_134 = arith.constant 8 : i32
        %pad1_134 = arith.constant 1 : i32
        %outidx_134 = arith.constant 134 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_134, %smean_134, %sscale_134, %dilation_134, %padding_134, %ki0_134, %ki1_134, %ki2_134, %pad1_134, %out_v, %outidx_134) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 135: dilation=1 padding=4 kernel=(3,4,8) padding1=1
        %fidx_135 = arith.constant 135 : index
        %bidx_135 = arith.constant 386 : index
        %smidx_135 = arith.constant 806 : index
        %ssidx_135 = arith.constant 1226 : index
        %bias_135 = memref.load %in_v[%bidx_135] : memref<1511xi32>
        %smean_135 = memref.load %in_v[%smidx_135] : memref<1511xi32>
        %sscale_135 = memref.load %in_v[%ssidx_135] : memref<1511xi32>
        %dilation_135 = arith.constant 1 : i32
        %padding_135 = arith.constant 4 : i32
        %ki0_135 = arith.constant 3 : i32
        %ki1_135 = arith.constant 4 : i32
        %ki2_135 = arith.constant 8 : i32
        %pad1_135 = arith.constant 1 : i32
        %outidx_135 = arith.constant 135 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_135, %smean_135, %sscale_135, %dilation_135, %padding_135, %ki0_135, %ki1_135, %ki2_135, %pad1_135, %out_v, %outidx_135) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 136: dilation=1 padding=4 kernel=(3,5,6) padding1=0
        %fidx_136 = arith.constant 136 : index
        %bidx_136 = arith.constant 387 : index
        %smidx_136 = arith.constant 807 : index
        %ssidx_136 = arith.constant 1227 : index
        %bias_136 = memref.load %in_v[%bidx_136] : memref<1511xi32>
        %smean_136 = memref.load %in_v[%smidx_136] : memref<1511xi32>
        %sscale_136 = memref.load %in_v[%ssidx_136] : memref<1511xi32>
        %dilation_136 = arith.constant 1 : i32
        %padding_136 = arith.constant 4 : i32
        %ki0_136 = arith.constant 3 : i32
        %ki1_136 = arith.constant 5 : i32
        %ki2_136 = arith.constant 6 : i32
        %pad1_136 = arith.constant 0 : i32
        %outidx_136 = arith.constant 136 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_136, %smean_136, %sscale_136, %dilation_136, %padding_136, %ki0_136, %ki1_136, %ki2_136, %pad1_136, %out_v, %outidx_136) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 137: dilation=1 padding=4 kernel=(3,5,6) padding1=0
        %fidx_137 = arith.constant 137 : index
        %bidx_137 = arith.constant 388 : index
        %smidx_137 = arith.constant 808 : index
        %ssidx_137 = arith.constant 1228 : index
        %bias_137 = memref.load %in_v[%bidx_137] : memref<1511xi32>
        %smean_137 = memref.load %in_v[%smidx_137] : memref<1511xi32>
        %sscale_137 = memref.load %in_v[%ssidx_137] : memref<1511xi32>
        %dilation_137 = arith.constant 1 : i32
        %padding_137 = arith.constant 4 : i32
        %ki0_137 = arith.constant 3 : i32
        %ki1_137 = arith.constant 5 : i32
        %ki2_137 = arith.constant 6 : i32
        %pad1_137 = arith.constant 0 : i32
        %outidx_137 = arith.constant 137 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_137, %smean_137, %sscale_137, %dilation_137, %padding_137, %ki0_137, %ki1_137, %ki2_137, %pad1_137, %out_v, %outidx_137) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 138: dilation=1 padding=4 kernel=(3,5,7) padding1=1
        %fidx_138 = arith.constant 138 : index
        %bidx_138 = arith.constant 389 : index
        %smidx_138 = arith.constant 809 : index
        %ssidx_138 = arith.constant 1229 : index
        %bias_138 = memref.load %in_v[%bidx_138] : memref<1511xi32>
        %smean_138 = memref.load %in_v[%smidx_138] : memref<1511xi32>
        %sscale_138 = memref.load %in_v[%ssidx_138] : memref<1511xi32>
        %dilation_138 = arith.constant 1 : i32
        %padding_138 = arith.constant 4 : i32
        %ki0_138 = arith.constant 3 : i32
        %ki1_138 = arith.constant 5 : i32
        %ki2_138 = arith.constant 7 : i32
        %pad1_138 = arith.constant 1 : i32
        %outidx_138 = arith.constant 138 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_138, %smean_138, %sscale_138, %dilation_138, %padding_138, %ki0_138, %ki1_138, %ki2_138, %pad1_138, %out_v, %outidx_138) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 139: dilation=1 padding=4 kernel=(3,5,7) padding1=1
        %fidx_139 = arith.constant 139 : index
        %bidx_139 = arith.constant 390 : index
        %smidx_139 = arith.constant 810 : index
        %ssidx_139 = arith.constant 1230 : index
        %bias_139 = memref.load %in_v[%bidx_139] : memref<1511xi32>
        %smean_139 = memref.load %in_v[%smidx_139] : memref<1511xi32>
        %sscale_139 = memref.load %in_v[%ssidx_139] : memref<1511xi32>
        %dilation_139 = arith.constant 1 : i32
        %padding_139 = arith.constant 4 : i32
        %ki0_139 = arith.constant 3 : i32
        %ki1_139 = arith.constant 5 : i32
        %ki2_139 = arith.constant 7 : i32
        %pad1_139 = arith.constant 1 : i32
        %outidx_139 = arith.constant 139 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_139, %smean_139, %sscale_139, %dilation_139, %padding_139, %ki0_139, %ki1_139, %ki2_139, %pad1_139, %out_v, %outidx_139) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 140: dilation=1 padding=4 kernel=(3,5,8) padding1=0
        %fidx_140 = arith.constant 140 : index
        %bidx_140 = arith.constant 391 : index
        %smidx_140 = arith.constant 811 : index
        %ssidx_140 = arith.constant 1231 : index
        %bias_140 = memref.load %in_v[%bidx_140] : memref<1511xi32>
        %smean_140 = memref.load %in_v[%smidx_140] : memref<1511xi32>
        %sscale_140 = memref.load %in_v[%ssidx_140] : memref<1511xi32>
        %dilation_140 = arith.constant 1 : i32
        %padding_140 = arith.constant 4 : i32
        %ki0_140 = arith.constant 3 : i32
        %ki1_140 = arith.constant 5 : i32
        %ki2_140 = arith.constant 8 : i32
        %pad1_140 = arith.constant 0 : i32
        %outidx_140 = arith.constant 140 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_140, %smean_140, %sscale_140, %dilation_140, %padding_140, %ki0_140, %ki1_140, %ki2_140, %pad1_140, %out_v, %outidx_140) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 141: dilation=1 padding=4 kernel=(3,5,8) padding1=0
        %fidx_141 = arith.constant 141 : index
        %bidx_141 = arith.constant 392 : index
        %smidx_141 = arith.constant 812 : index
        %ssidx_141 = arith.constant 1232 : index
        %bias_141 = memref.load %in_v[%bidx_141] : memref<1511xi32>
        %smean_141 = memref.load %in_v[%smidx_141] : memref<1511xi32>
        %sscale_141 = memref.load %in_v[%ssidx_141] : memref<1511xi32>
        %dilation_141 = arith.constant 1 : i32
        %padding_141 = arith.constant 4 : i32
        %ki0_141 = arith.constant 3 : i32
        %ki1_141 = arith.constant 5 : i32
        %ki2_141 = arith.constant 8 : i32
        %pad1_141 = arith.constant 0 : i32
        %outidx_141 = arith.constant 141 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_141, %smean_141, %sscale_141, %dilation_141, %padding_141, %ki0_141, %ki1_141, %ki2_141, %pad1_141, %out_v, %outidx_141) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 142: dilation=1 padding=4 kernel=(3,6,7) padding1=1
        %fidx_142 = arith.constant 142 : index
        %bidx_142 = arith.constant 393 : index
        %smidx_142 = arith.constant 813 : index
        %ssidx_142 = arith.constant 1233 : index
        %bias_142 = memref.load %in_v[%bidx_142] : memref<1511xi32>
        %smean_142 = memref.load %in_v[%smidx_142] : memref<1511xi32>
        %sscale_142 = memref.load %in_v[%ssidx_142] : memref<1511xi32>
        %dilation_142 = arith.constant 1 : i32
        %padding_142 = arith.constant 4 : i32
        %ki0_142 = arith.constant 3 : i32
        %ki1_142 = arith.constant 6 : i32
        %ki2_142 = arith.constant 7 : i32
        %pad1_142 = arith.constant 1 : i32
        %outidx_142 = arith.constant 142 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_142, %smean_142, %sscale_142, %dilation_142, %padding_142, %ki0_142, %ki1_142, %ki2_142, %pad1_142, %out_v, %outidx_142) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 143: dilation=1 padding=4 kernel=(3,6,7) padding1=1
        %fidx_143 = arith.constant 143 : index
        %bidx_143 = arith.constant 394 : index
        %smidx_143 = arith.constant 814 : index
        %ssidx_143 = arith.constant 1234 : index
        %bias_143 = memref.load %in_v[%bidx_143] : memref<1511xi32>
        %smean_143 = memref.load %in_v[%smidx_143] : memref<1511xi32>
        %sscale_143 = memref.load %in_v[%ssidx_143] : memref<1511xi32>
        %dilation_143 = arith.constant 1 : i32
        %padding_143 = arith.constant 4 : i32
        %ki0_143 = arith.constant 3 : i32
        %ki1_143 = arith.constant 6 : i32
        %ki2_143 = arith.constant 7 : i32
        %pad1_143 = arith.constant 1 : i32
        %outidx_143 = arith.constant 143 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_143, %smean_143, %sscale_143, %dilation_143, %padding_143, %ki0_143, %ki1_143, %ki2_143, %pad1_143, %out_v, %outidx_143) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 144: dilation=1 padding=4 kernel=(3,6,8) padding1=0
        %fidx_144 = arith.constant 144 : index
        %bidx_144 = arith.constant 395 : index
        %smidx_144 = arith.constant 815 : index
        %ssidx_144 = arith.constant 1235 : index
        %bias_144 = memref.load %in_v[%bidx_144] : memref<1511xi32>
        %smean_144 = memref.load %in_v[%smidx_144] : memref<1511xi32>
        %sscale_144 = memref.load %in_v[%ssidx_144] : memref<1511xi32>
        %dilation_144 = arith.constant 1 : i32
        %padding_144 = arith.constant 4 : i32
        %ki0_144 = arith.constant 3 : i32
        %ki1_144 = arith.constant 6 : i32
        %ki2_144 = arith.constant 8 : i32
        %pad1_144 = arith.constant 0 : i32
        %outidx_144 = arith.constant 144 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_144, %smean_144, %sscale_144, %dilation_144, %padding_144, %ki0_144, %ki1_144, %ki2_144, %pad1_144, %out_v, %outidx_144) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 145: dilation=1 padding=4 kernel=(3,6,8) padding1=0
        %fidx_145 = arith.constant 145 : index
        %bidx_145 = arith.constant 396 : index
        %smidx_145 = arith.constant 816 : index
        %ssidx_145 = arith.constant 1236 : index
        %bias_145 = memref.load %in_v[%bidx_145] : memref<1511xi32>
        %smean_145 = memref.load %in_v[%smidx_145] : memref<1511xi32>
        %sscale_145 = memref.load %in_v[%ssidx_145] : memref<1511xi32>
        %dilation_145 = arith.constant 1 : i32
        %padding_145 = arith.constant 4 : i32
        %ki0_145 = arith.constant 3 : i32
        %ki1_145 = arith.constant 6 : i32
        %ki2_145 = arith.constant 8 : i32
        %pad1_145 = arith.constant 0 : i32
        %outidx_145 = arith.constant 145 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_145, %smean_145, %sscale_145, %dilation_145, %padding_145, %ki0_145, %ki1_145, %ki2_145, %pad1_145, %out_v, %outidx_145) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 146: dilation=1 padding=4 kernel=(3,7,8) padding1=1
        %fidx_146 = arith.constant 146 : index
        %bidx_146 = arith.constant 397 : index
        %smidx_146 = arith.constant 817 : index
        %ssidx_146 = arith.constant 1237 : index
        %bias_146 = memref.load %in_v[%bidx_146] : memref<1511xi32>
        %smean_146 = memref.load %in_v[%smidx_146] : memref<1511xi32>
        %sscale_146 = memref.load %in_v[%ssidx_146] : memref<1511xi32>
        %dilation_146 = arith.constant 1 : i32
        %padding_146 = arith.constant 4 : i32
        %ki0_146 = arith.constant 3 : i32
        %ki1_146 = arith.constant 7 : i32
        %ki2_146 = arith.constant 8 : i32
        %pad1_146 = arith.constant 1 : i32
        %outidx_146 = arith.constant 146 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_146, %smean_146, %sscale_146, %dilation_146, %padding_146, %ki0_146, %ki1_146, %ki2_146, %pad1_146, %out_v, %outidx_146) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 147: dilation=1 padding=4 kernel=(3,7,8) padding1=1
        %fidx_147 = arith.constant 147 : index
        %bidx_147 = arith.constant 398 : index
        %smidx_147 = arith.constant 818 : index
        %ssidx_147 = arith.constant 1238 : index
        %bias_147 = memref.load %in_v[%bidx_147] : memref<1511xi32>
        %smean_147 = memref.load %in_v[%smidx_147] : memref<1511xi32>
        %sscale_147 = memref.load %in_v[%ssidx_147] : memref<1511xi32>
        %dilation_147 = arith.constant 1 : i32
        %padding_147 = arith.constant 4 : i32
        %ki0_147 = arith.constant 3 : i32
        %ki1_147 = arith.constant 7 : i32
        %ki2_147 = arith.constant 8 : i32
        %pad1_147 = arith.constant 1 : i32
        %outidx_147 = arith.constant 147 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_147, %smean_147, %sscale_147, %dilation_147, %padding_147, %ki0_147, %ki1_147, %ki2_147, %pad1_147, %out_v, %outidx_147) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 148: dilation=1 padding=4 kernel=(4,5,6) padding1=0
        %fidx_148 = arith.constant 148 : index
        %bidx_148 = arith.constant 399 : index
        %smidx_148 = arith.constant 819 : index
        %ssidx_148 = arith.constant 1239 : index
        %bias_148 = memref.load %in_v[%bidx_148] : memref<1511xi32>
        %smean_148 = memref.load %in_v[%smidx_148] : memref<1511xi32>
        %sscale_148 = memref.load %in_v[%ssidx_148] : memref<1511xi32>
        %dilation_148 = arith.constant 1 : i32
        %padding_148 = arith.constant 4 : i32
        %ki0_148 = arith.constant 4 : i32
        %ki1_148 = arith.constant 5 : i32
        %ki2_148 = arith.constant 6 : i32
        %pad1_148 = arith.constant 0 : i32
        %outidx_148 = arith.constant 148 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_148, %smean_148, %sscale_148, %dilation_148, %padding_148, %ki0_148, %ki1_148, %ki2_148, %pad1_148, %out_v, %outidx_148) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 149: dilation=1 padding=4 kernel=(4,5,6) padding1=0
        %fidx_149 = arith.constant 149 : index
        %bidx_149 = arith.constant 400 : index
        %smidx_149 = arith.constant 820 : index
        %ssidx_149 = arith.constant 1240 : index
        %bias_149 = memref.load %in_v[%bidx_149] : memref<1511xi32>
        %smean_149 = memref.load %in_v[%smidx_149] : memref<1511xi32>
        %sscale_149 = memref.load %in_v[%ssidx_149] : memref<1511xi32>
        %dilation_149 = arith.constant 1 : i32
        %padding_149 = arith.constant 4 : i32
        %ki0_149 = arith.constant 4 : i32
        %ki1_149 = arith.constant 5 : i32
        %ki2_149 = arith.constant 6 : i32
        %pad1_149 = arith.constant 0 : i32
        %outidx_149 = arith.constant 149 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_149, %smean_149, %sscale_149, %dilation_149, %padding_149, %ki0_149, %ki1_149, %ki2_149, %pad1_149, %out_v, %outidx_149) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 150: dilation=1 padding=4 kernel=(4,5,7) padding1=1
        %fidx_150 = arith.constant 150 : index
        %bidx_150 = arith.constant 401 : index
        %smidx_150 = arith.constant 821 : index
        %ssidx_150 = arith.constant 1241 : index
        %bias_150 = memref.load %in_v[%bidx_150] : memref<1511xi32>
        %smean_150 = memref.load %in_v[%smidx_150] : memref<1511xi32>
        %sscale_150 = memref.load %in_v[%ssidx_150] : memref<1511xi32>
        %dilation_150 = arith.constant 1 : i32
        %padding_150 = arith.constant 4 : i32
        %ki0_150 = arith.constant 4 : i32
        %ki1_150 = arith.constant 5 : i32
        %ki2_150 = arith.constant 7 : i32
        %pad1_150 = arith.constant 1 : i32
        %outidx_150 = arith.constant 150 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_150, %smean_150, %sscale_150, %dilation_150, %padding_150, %ki0_150, %ki1_150, %ki2_150, %pad1_150, %out_v, %outidx_150) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 151: dilation=1 padding=4 kernel=(4,5,7) padding1=1
        %fidx_151 = arith.constant 151 : index
        %bidx_151 = arith.constant 402 : index
        %smidx_151 = arith.constant 822 : index
        %ssidx_151 = arith.constant 1242 : index
        %bias_151 = memref.load %in_v[%bidx_151] : memref<1511xi32>
        %smean_151 = memref.load %in_v[%smidx_151] : memref<1511xi32>
        %sscale_151 = memref.load %in_v[%ssidx_151] : memref<1511xi32>
        %dilation_151 = arith.constant 1 : i32
        %padding_151 = arith.constant 4 : i32
        %ki0_151 = arith.constant 4 : i32
        %ki1_151 = arith.constant 5 : i32
        %ki2_151 = arith.constant 7 : i32
        %pad1_151 = arith.constant 1 : i32
        %outidx_151 = arith.constant 151 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_151, %smean_151, %sscale_151, %dilation_151, %padding_151, %ki0_151, %ki1_151, %ki2_151, %pad1_151, %out_v, %outidx_151) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 152: dilation=1 padding=4 kernel=(4,5,8) padding1=0
        %fidx_152 = arith.constant 152 : index
        %bidx_152 = arith.constant 403 : index
        %smidx_152 = arith.constant 823 : index
        %ssidx_152 = arith.constant 1243 : index
        %bias_152 = memref.load %in_v[%bidx_152] : memref<1511xi32>
        %smean_152 = memref.load %in_v[%smidx_152] : memref<1511xi32>
        %sscale_152 = memref.load %in_v[%ssidx_152] : memref<1511xi32>
        %dilation_152 = arith.constant 1 : i32
        %padding_152 = arith.constant 4 : i32
        %ki0_152 = arith.constant 4 : i32
        %ki1_152 = arith.constant 5 : i32
        %ki2_152 = arith.constant 8 : i32
        %pad1_152 = arith.constant 0 : i32
        %outidx_152 = arith.constant 152 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_152, %smean_152, %sscale_152, %dilation_152, %padding_152, %ki0_152, %ki1_152, %ki2_152, %pad1_152, %out_v, %outidx_152) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 153: dilation=1 padding=4 kernel=(4,5,8) padding1=0
        %fidx_153 = arith.constant 153 : index
        %bidx_153 = arith.constant 404 : index
        %smidx_153 = arith.constant 824 : index
        %ssidx_153 = arith.constant 1244 : index
        %bias_153 = memref.load %in_v[%bidx_153] : memref<1511xi32>
        %smean_153 = memref.load %in_v[%smidx_153] : memref<1511xi32>
        %sscale_153 = memref.load %in_v[%ssidx_153] : memref<1511xi32>
        %dilation_153 = arith.constant 1 : i32
        %padding_153 = arith.constant 4 : i32
        %ki0_153 = arith.constant 4 : i32
        %ki1_153 = arith.constant 5 : i32
        %ki2_153 = arith.constant 8 : i32
        %pad1_153 = arith.constant 0 : i32
        %outidx_153 = arith.constant 153 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_153, %smean_153, %sscale_153, %dilation_153, %padding_153, %ki0_153, %ki1_153, %ki2_153, %pad1_153, %out_v, %outidx_153) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 154: dilation=1 padding=4 kernel=(4,6,7) padding1=1
        %fidx_154 = arith.constant 154 : index
        %bidx_154 = arith.constant 405 : index
        %smidx_154 = arith.constant 825 : index
        %ssidx_154 = arith.constant 1245 : index
        %bias_154 = memref.load %in_v[%bidx_154] : memref<1511xi32>
        %smean_154 = memref.load %in_v[%smidx_154] : memref<1511xi32>
        %sscale_154 = memref.load %in_v[%ssidx_154] : memref<1511xi32>
        %dilation_154 = arith.constant 1 : i32
        %padding_154 = arith.constant 4 : i32
        %ki0_154 = arith.constant 4 : i32
        %ki1_154 = arith.constant 6 : i32
        %ki2_154 = arith.constant 7 : i32
        %pad1_154 = arith.constant 1 : i32
        %outidx_154 = arith.constant 154 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_154, %smean_154, %sscale_154, %dilation_154, %padding_154, %ki0_154, %ki1_154, %ki2_154, %pad1_154, %out_v, %outidx_154) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 155: dilation=1 padding=4 kernel=(4,6,7) padding1=1
        %fidx_155 = arith.constant 155 : index
        %bidx_155 = arith.constant 406 : index
        %smidx_155 = arith.constant 826 : index
        %ssidx_155 = arith.constant 1246 : index
        %bias_155 = memref.load %in_v[%bidx_155] : memref<1511xi32>
        %smean_155 = memref.load %in_v[%smidx_155] : memref<1511xi32>
        %sscale_155 = memref.load %in_v[%ssidx_155] : memref<1511xi32>
        %dilation_155 = arith.constant 1 : i32
        %padding_155 = arith.constant 4 : i32
        %ki0_155 = arith.constant 4 : i32
        %ki1_155 = arith.constant 6 : i32
        %ki2_155 = arith.constant 7 : i32
        %pad1_155 = arith.constant 1 : i32
        %outidx_155 = arith.constant 155 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_155, %smean_155, %sscale_155, %dilation_155, %padding_155, %ki0_155, %ki1_155, %ki2_155, %pad1_155, %out_v, %outidx_155) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 156: dilation=1 padding=4 kernel=(4,6,8) padding1=0
        %fidx_156 = arith.constant 156 : index
        %bidx_156 = arith.constant 407 : index
        %smidx_156 = arith.constant 827 : index
        %ssidx_156 = arith.constant 1247 : index
        %bias_156 = memref.load %in_v[%bidx_156] : memref<1511xi32>
        %smean_156 = memref.load %in_v[%smidx_156] : memref<1511xi32>
        %sscale_156 = memref.load %in_v[%ssidx_156] : memref<1511xi32>
        %dilation_156 = arith.constant 1 : i32
        %padding_156 = arith.constant 4 : i32
        %ki0_156 = arith.constant 4 : i32
        %ki1_156 = arith.constant 6 : i32
        %ki2_156 = arith.constant 8 : i32
        %pad1_156 = arith.constant 0 : i32
        %outidx_156 = arith.constant 156 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_156, %smean_156, %sscale_156, %dilation_156, %padding_156, %ki0_156, %ki1_156, %ki2_156, %pad1_156, %out_v, %outidx_156) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 157: dilation=1 padding=4 kernel=(4,6,8) padding1=0
        %fidx_157 = arith.constant 157 : index
        %bidx_157 = arith.constant 408 : index
        %smidx_157 = arith.constant 828 : index
        %ssidx_157 = arith.constant 1248 : index
        %bias_157 = memref.load %in_v[%bidx_157] : memref<1511xi32>
        %smean_157 = memref.load %in_v[%smidx_157] : memref<1511xi32>
        %sscale_157 = memref.load %in_v[%ssidx_157] : memref<1511xi32>
        %dilation_157 = arith.constant 1 : i32
        %padding_157 = arith.constant 4 : i32
        %ki0_157 = arith.constant 4 : i32
        %ki1_157 = arith.constant 6 : i32
        %ki2_157 = arith.constant 8 : i32
        %pad1_157 = arith.constant 0 : i32
        %outidx_157 = arith.constant 157 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_157, %smean_157, %sscale_157, %dilation_157, %padding_157, %ki0_157, %ki1_157, %ki2_157, %pad1_157, %out_v, %outidx_157) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 158: dilation=1 padding=4 kernel=(4,7,8) padding1=1
        %fidx_158 = arith.constant 158 : index
        %bidx_158 = arith.constant 409 : index
        %smidx_158 = arith.constant 829 : index
        %ssidx_158 = arith.constant 1249 : index
        %bias_158 = memref.load %in_v[%bidx_158] : memref<1511xi32>
        %smean_158 = memref.load %in_v[%smidx_158] : memref<1511xi32>
        %sscale_158 = memref.load %in_v[%ssidx_158] : memref<1511xi32>
        %dilation_158 = arith.constant 1 : i32
        %padding_158 = arith.constant 4 : i32
        %ki0_158 = arith.constant 4 : i32
        %ki1_158 = arith.constant 7 : i32
        %ki2_158 = arith.constant 8 : i32
        %pad1_158 = arith.constant 1 : i32
        %outidx_158 = arith.constant 158 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_158, %smean_158, %sscale_158, %dilation_158, %padding_158, %ki0_158, %ki1_158, %ki2_158, %pad1_158, %out_v, %outidx_158) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 159: dilation=1 padding=4 kernel=(4,7,8) padding1=1
        %fidx_159 = arith.constant 159 : index
        %bidx_159 = arith.constant 410 : index
        %smidx_159 = arith.constant 830 : index
        %ssidx_159 = arith.constant 1250 : index
        %bias_159 = memref.load %in_v[%bidx_159] : memref<1511xi32>
        %smean_159 = memref.load %in_v[%smidx_159] : memref<1511xi32>
        %sscale_159 = memref.load %in_v[%ssidx_159] : memref<1511xi32>
        %dilation_159 = arith.constant 1 : i32
        %padding_159 = arith.constant 4 : i32
        %ki0_159 = arith.constant 4 : i32
        %ki1_159 = arith.constant 7 : i32
        %ki2_159 = arith.constant 8 : i32
        %pad1_159 = arith.constant 1 : i32
        %outidx_159 = arith.constant 159 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_159, %smean_159, %sscale_159, %dilation_159, %padding_159, %ki0_159, %ki1_159, %ki2_159, %pad1_159, %out_v, %outidx_159) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 160: dilation=1 padding=4 kernel=(5,6,7) padding1=0
        %fidx_160 = arith.constant 160 : index
        %bidx_160 = arith.constant 411 : index
        %smidx_160 = arith.constant 831 : index
        %ssidx_160 = arith.constant 1251 : index
        %bias_160 = memref.load %in_v[%bidx_160] : memref<1511xi32>
        %smean_160 = memref.load %in_v[%smidx_160] : memref<1511xi32>
        %sscale_160 = memref.load %in_v[%ssidx_160] : memref<1511xi32>
        %dilation_160 = arith.constant 1 : i32
        %padding_160 = arith.constant 4 : i32
        %ki0_160 = arith.constant 5 : i32
        %ki1_160 = arith.constant 6 : i32
        %ki2_160 = arith.constant 7 : i32
        %pad1_160 = arith.constant 0 : i32
        %outidx_160 = arith.constant 160 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_160, %smean_160, %sscale_160, %dilation_160, %padding_160, %ki0_160, %ki1_160, %ki2_160, %pad1_160, %out_v, %outidx_160) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 161: dilation=1 padding=4 kernel=(5,6,7) padding1=0
        %fidx_161 = arith.constant 161 : index
        %bidx_161 = arith.constant 412 : index
        %smidx_161 = arith.constant 832 : index
        %ssidx_161 = arith.constant 1252 : index
        %bias_161 = memref.load %in_v[%bidx_161] : memref<1511xi32>
        %smean_161 = memref.load %in_v[%smidx_161] : memref<1511xi32>
        %sscale_161 = memref.load %in_v[%ssidx_161] : memref<1511xi32>
        %dilation_161 = arith.constant 1 : i32
        %padding_161 = arith.constant 4 : i32
        %ki0_161 = arith.constant 5 : i32
        %ki1_161 = arith.constant 6 : i32
        %ki2_161 = arith.constant 7 : i32
        %pad1_161 = arith.constant 0 : i32
        %outidx_161 = arith.constant 161 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_161, %smean_161, %sscale_161, %dilation_161, %padding_161, %ki0_161, %ki1_161, %ki2_161, %pad1_161, %out_v, %outidx_161) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 162: dilation=1 padding=4 kernel=(5,6,8) padding1=1
        %fidx_162 = arith.constant 162 : index
        %bidx_162 = arith.constant 413 : index
        %smidx_162 = arith.constant 833 : index
        %ssidx_162 = arith.constant 1253 : index
        %bias_162 = memref.load %in_v[%bidx_162] : memref<1511xi32>
        %smean_162 = memref.load %in_v[%smidx_162] : memref<1511xi32>
        %sscale_162 = memref.load %in_v[%ssidx_162] : memref<1511xi32>
        %dilation_162 = arith.constant 1 : i32
        %padding_162 = arith.constant 4 : i32
        %ki0_162 = arith.constant 5 : i32
        %ki1_162 = arith.constant 6 : i32
        %ki2_162 = arith.constant 8 : i32
        %pad1_162 = arith.constant 1 : i32
        %outidx_162 = arith.constant 162 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_162, %smean_162, %sscale_162, %dilation_162, %padding_162, %ki0_162, %ki1_162, %ki2_162, %pad1_162, %out_v, %outidx_162) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 163: dilation=1 padding=4 kernel=(5,6,8) padding1=1
        %fidx_163 = arith.constant 163 : index
        %bidx_163 = arith.constant 414 : index
        %smidx_163 = arith.constant 834 : index
        %ssidx_163 = arith.constant 1254 : index
        %bias_163 = memref.load %in_v[%bidx_163] : memref<1511xi32>
        %smean_163 = memref.load %in_v[%smidx_163] : memref<1511xi32>
        %sscale_163 = memref.load %in_v[%ssidx_163] : memref<1511xi32>
        %dilation_163 = arith.constant 1 : i32
        %padding_163 = arith.constant 4 : i32
        %ki0_163 = arith.constant 5 : i32
        %ki1_163 = arith.constant 6 : i32
        %ki2_163 = arith.constant 8 : i32
        %pad1_163 = arith.constant 1 : i32
        %outidx_163 = arith.constant 163 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_163, %smean_163, %sscale_163, %dilation_163, %padding_163, %ki0_163, %ki1_163, %ki2_163, %pad1_163, %out_v, %outidx_163) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 164: dilation=1 padding=4 kernel=(5,7,8) padding1=0
        %fidx_164 = arith.constant 164 : index
        %bidx_164 = arith.constant 415 : index
        %smidx_164 = arith.constant 835 : index
        %ssidx_164 = arith.constant 1255 : index
        %bias_164 = memref.load %in_v[%bidx_164] : memref<1511xi32>
        %smean_164 = memref.load %in_v[%smidx_164] : memref<1511xi32>
        %sscale_164 = memref.load %in_v[%ssidx_164] : memref<1511xi32>
        %dilation_164 = arith.constant 1 : i32
        %padding_164 = arith.constant 4 : i32
        %ki0_164 = arith.constant 5 : i32
        %ki1_164 = arith.constant 7 : i32
        %ki2_164 = arith.constant 8 : i32
        %pad1_164 = arith.constant 0 : i32
        %outidx_164 = arith.constant 164 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_164, %smean_164, %sscale_164, %dilation_164, %padding_164, %ki0_164, %ki1_164, %ki2_164, %pad1_164, %out_v, %outidx_164) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 165: dilation=1 padding=4 kernel=(5,7,8) padding1=0
        %fidx_165 = arith.constant 165 : index
        %bidx_165 = arith.constant 416 : index
        %smidx_165 = arith.constant 836 : index
        %ssidx_165 = arith.constant 1256 : index
        %bias_165 = memref.load %in_v[%bidx_165] : memref<1511xi32>
        %smean_165 = memref.load %in_v[%smidx_165] : memref<1511xi32>
        %sscale_165 = memref.load %in_v[%ssidx_165] : memref<1511xi32>
        %dilation_165 = arith.constant 1 : i32
        %padding_165 = arith.constant 4 : i32
        %ki0_165 = arith.constant 5 : i32
        %ki1_165 = arith.constant 7 : i32
        %ki2_165 = arith.constant 8 : i32
        %pad1_165 = arith.constant 0 : i32
        %outidx_165 = arith.constant 165 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_165, %smean_165, %sscale_165, %dilation_165, %padding_165, %ki0_165, %ki1_165, %ki2_165, %pad1_165, %out_v, %outidx_165) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 166: dilation=1 padding=4 kernel=(6,7,8) padding1=1
        %fidx_166 = arith.constant 166 : index
        %bidx_166 = arith.constant 417 : index
        %smidx_166 = arith.constant 837 : index
        %ssidx_166 = arith.constant 1257 : index
        %bias_166 = memref.load %in_v[%bidx_166] : memref<1511xi32>
        %smean_166 = memref.load %in_v[%smidx_166] : memref<1511xi32>
        %sscale_166 = memref.load %in_v[%ssidx_166] : memref<1511xi32>
        %dilation_166 = arith.constant 1 : i32
        %padding_166 = arith.constant 4 : i32
        %ki0_166 = arith.constant 6 : i32
        %ki1_166 = arith.constant 7 : i32
        %ki2_166 = arith.constant 8 : i32
        %pad1_166 = arith.constant 1 : i32
        %outidx_166 = arith.constant 166 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_166, %smean_166, %sscale_166, %dilation_166, %padding_166, %ki0_166, %ki1_166, %ki2_166, %pad1_166, %out_v, %outidx_166) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 167: dilation=1 padding=4 kernel=(6,7,8) padding1=1
        %fidx_167 = arith.constant 167 : index
        %bidx_167 = arith.constant 418 : index
        %smidx_167 = arith.constant 838 : index
        %ssidx_167 = arith.constant 1258 : index
        %bias_167 = memref.load %in_v[%bidx_167] : memref<1511xi32>
        %smean_167 = memref.load %in_v[%smidx_167] : memref<1511xi32>
        %sscale_167 = memref.load %in_v[%ssidx_167] : memref<1511xi32>
        %dilation_167 = arith.constant 1 : i32
        %padding_167 = arith.constant 4 : i32
        %ki0_167 = arith.constant 6 : i32
        %ki1_167 = arith.constant 7 : i32
        %ki2_167 = arith.constant 8 : i32
        %pad1_167 = arith.constant 1 : i32
        %outidx_167 = arith.constant 167 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_167, %smean_167, %sscale_167, %dilation_167, %padding_167, %ki0_167, %ki1_167, %ki2_167, %pad1_167, %out_v, %outidx_167) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 168: dilation=2 padding=8 kernel=(0,1,2) padding1=1
        %fidx_168 = arith.constant 168 : index
        %bidx_168 = arith.constant 419 : index
        %smidx_168 = arith.constant 839 : index
        %ssidx_168 = arith.constant 1259 : index
        %bias_168 = memref.load %in_v[%bidx_168] : memref<1511xi32>
        %smean_168 = memref.load %in_v[%smidx_168] : memref<1511xi32>
        %sscale_168 = memref.load %in_v[%ssidx_168] : memref<1511xi32>
        %dilation_168 = arith.constant 2 : i32
        %padding_168 = arith.constant 8 : i32
        %ki0_168 = arith.constant 0 : i32
        %ki1_168 = arith.constant 1 : i32
        %ki2_168 = arith.constant 2 : i32
        %pad1_168 = arith.constant 1 : i32
        %outidx_168 = arith.constant 168 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_168, %smean_168, %sscale_168, %dilation_168, %padding_168, %ki0_168, %ki1_168, %ki2_168, %pad1_168, %out_v, %outidx_168) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 169: dilation=2 padding=8 kernel=(0,1,3) padding1=0
        %fidx_169 = arith.constant 169 : index
        %bidx_169 = arith.constant 420 : index
        %smidx_169 = arith.constant 840 : index
        %ssidx_169 = arith.constant 1260 : index
        %bias_169 = memref.load %in_v[%bidx_169] : memref<1511xi32>
        %smean_169 = memref.load %in_v[%smidx_169] : memref<1511xi32>
        %sscale_169 = memref.load %in_v[%ssidx_169] : memref<1511xi32>
        %dilation_169 = arith.constant 2 : i32
        %padding_169 = arith.constant 8 : i32
        %ki0_169 = arith.constant 0 : i32
        %ki1_169 = arith.constant 1 : i32
        %ki2_169 = arith.constant 3 : i32
        %pad1_169 = arith.constant 0 : i32
        %outidx_169 = arith.constant 169 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_169, %smean_169, %sscale_169, %dilation_169, %padding_169, %ki0_169, %ki1_169, %ki2_169, %pad1_169, %out_v, %outidx_169) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 170: dilation=2 padding=8 kernel=(0,1,4) padding1=1
        %fidx_170 = arith.constant 170 : index
        %bidx_170 = arith.constant 421 : index
        %smidx_170 = arith.constant 841 : index
        %ssidx_170 = arith.constant 1261 : index
        %bias_170 = memref.load %in_v[%bidx_170] : memref<1511xi32>
        %smean_170 = memref.load %in_v[%smidx_170] : memref<1511xi32>
        %sscale_170 = memref.load %in_v[%ssidx_170] : memref<1511xi32>
        %dilation_170 = arith.constant 2 : i32
        %padding_170 = arith.constant 8 : i32
        %ki0_170 = arith.constant 0 : i32
        %ki1_170 = arith.constant 1 : i32
        %ki2_170 = arith.constant 4 : i32
        %pad1_170 = arith.constant 1 : i32
        %outidx_170 = arith.constant 170 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_170, %smean_170, %sscale_170, %dilation_170, %padding_170, %ki0_170, %ki1_170, %ki2_170, %pad1_170, %out_v, %outidx_170) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 171: dilation=2 padding=8 kernel=(0,1,5) padding1=0
        %fidx_171 = arith.constant 171 : index
        %bidx_171 = arith.constant 422 : index
        %smidx_171 = arith.constant 842 : index
        %ssidx_171 = arith.constant 1262 : index
        %bias_171 = memref.load %in_v[%bidx_171] : memref<1511xi32>
        %smean_171 = memref.load %in_v[%smidx_171] : memref<1511xi32>
        %sscale_171 = memref.load %in_v[%ssidx_171] : memref<1511xi32>
        %dilation_171 = arith.constant 2 : i32
        %padding_171 = arith.constant 8 : i32
        %ki0_171 = arith.constant 0 : i32
        %ki1_171 = arith.constant 1 : i32
        %ki2_171 = arith.constant 5 : i32
        %pad1_171 = arith.constant 0 : i32
        %outidx_171 = arith.constant 171 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_171, %smean_171, %sscale_171, %dilation_171, %padding_171, %ki0_171, %ki1_171, %ki2_171, %pad1_171, %out_v, %outidx_171) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 172: dilation=2 padding=8 kernel=(0,1,6) padding1=1
        %fidx_172 = arith.constant 172 : index
        %bidx_172 = arith.constant 423 : index
        %smidx_172 = arith.constant 843 : index
        %ssidx_172 = arith.constant 1263 : index
        %bias_172 = memref.load %in_v[%bidx_172] : memref<1511xi32>
        %smean_172 = memref.load %in_v[%smidx_172] : memref<1511xi32>
        %sscale_172 = memref.load %in_v[%ssidx_172] : memref<1511xi32>
        %dilation_172 = arith.constant 2 : i32
        %padding_172 = arith.constant 8 : i32
        %ki0_172 = arith.constant 0 : i32
        %ki1_172 = arith.constant 1 : i32
        %ki2_172 = arith.constant 6 : i32
        %pad1_172 = arith.constant 1 : i32
        %outidx_172 = arith.constant 172 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_172, %smean_172, %sscale_172, %dilation_172, %padding_172, %ki0_172, %ki1_172, %ki2_172, %pad1_172, %out_v, %outidx_172) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 173: dilation=2 padding=8 kernel=(0,1,7) padding1=0
        %fidx_173 = arith.constant 173 : index
        %bidx_173 = arith.constant 424 : index
        %smidx_173 = arith.constant 844 : index
        %ssidx_173 = arith.constant 1264 : index
        %bias_173 = memref.load %in_v[%bidx_173] : memref<1511xi32>
        %smean_173 = memref.load %in_v[%smidx_173] : memref<1511xi32>
        %sscale_173 = memref.load %in_v[%ssidx_173] : memref<1511xi32>
        %dilation_173 = arith.constant 2 : i32
        %padding_173 = arith.constant 8 : i32
        %ki0_173 = arith.constant 0 : i32
        %ki1_173 = arith.constant 1 : i32
        %ki2_173 = arith.constant 7 : i32
        %pad1_173 = arith.constant 0 : i32
        %outidx_173 = arith.constant 173 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_173, %smean_173, %sscale_173, %dilation_173, %padding_173, %ki0_173, %ki1_173, %ki2_173, %pad1_173, %out_v, %outidx_173) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 174: dilation=2 padding=8 kernel=(0,1,8) padding1=1
        %fidx_174 = arith.constant 174 : index
        %bidx_174 = arith.constant 425 : index
        %smidx_174 = arith.constant 845 : index
        %ssidx_174 = arith.constant 1265 : index
        %bias_174 = memref.load %in_v[%bidx_174] : memref<1511xi32>
        %smean_174 = memref.load %in_v[%smidx_174] : memref<1511xi32>
        %sscale_174 = memref.load %in_v[%ssidx_174] : memref<1511xi32>
        %dilation_174 = arith.constant 2 : i32
        %padding_174 = arith.constant 8 : i32
        %ki0_174 = arith.constant 0 : i32
        %ki1_174 = arith.constant 1 : i32
        %ki2_174 = arith.constant 8 : i32
        %pad1_174 = arith.constant 1 : i32
        %outidx_174 = arith.constant 174 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_174, %smean_174, %sscale_174, %dilation_174, %padding_174, %ki0_174, %ki1_174, %ki2_174, %pad1_174, %out_v, %outidx_174) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 175: dilation=2 padding=8 kernel=(0,2,3) padding1=0
        %fidx_175 = arith.constant 175 : index
        %bidx_175 = arith.constant 426 : index
        %smidx_175 = arith.constant 846 : index
        %ssidx_175 = arith.constant 1266 : index
        %bias_175 = memref.load %in_v[%bidx_175] : memref<1511xi32>
        %smean_175 = memref.load %in_v[%smidx_175] : memref<1511xi32>
        %sscale_175 = memref.load %in_v[%ssidx_175] : memref<1511xi32>
        %dilation_175 = arith.constant 2 : i32
        %padding_175 = arith.constant 8 : i32
        %ki0_175 = arith.constant 0 : i32
        %ki1_175 = arith.constant 2 : i32
        %ki2_175 = arith.constant 3 : i32
        %pad1_175 = arith.constant 0 : i32
        %outidx_175 = arith.constant 175 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_175, %smean_175, %sscale_175, %dilation_175, %padding_175, %ki0_175, %ki1_175, %ki2_175, %pad1_175, %out_v, %outidx_175) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 176: dilation=2 padding=8 kernel=(0,2,4) padding1=1
        %fidx_176 = arith.constant 176 : index
        %bidx_176 = arith.constant 427 : index
        %smidx_176 = arith.constant 847 : index
        %ssidx_176 = arith.constant 1267 : index
        %bias_176 = memref.load %in_v[%bidx_176] : memref<1511xi32>
        %smean_176 = memref.load %in_v[%smidx_176] : memref<1511xi32>
        %sscale_176 = memref.load %in_v[%ssidx_176] : memref<1511xi32>
        %dilation_176 = arith.constant 2 : i32
        %padding_176 = arith.constant 8 : i32
        %ki0_176 = arith.constant 0 : i32
        %ki1_176 = arith.constant 2 : i32
        %ki2_176 = arith.constant 4 : i32
        %pad1_176 = arith.constant 1 : i32
        %outidx_176 = arith.constant 176 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_176, %smean_176, %sscale_176, %dilation_176, %padding_176, %ki0_176, %ki1_176, %ki2_176, %pad1_176, %out_v, %outidx_176) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 177: dilation=2 padding=8 kernel=(0,2,5) padding1=0
        %fidx_177 = arith.constant 177 : index
        %bidx_177 = arith.constant 428 : index
        %smidx_177 = arith.constant 848 : index
        %ssidx_177 = arith.constant 1268 : index
        %bias_177 = memref.load %in_v[%bidx_177] : memref<1511xi32>
        %smean_177 = memref.load %in_v[%smidx_177] : memref<1511xi32>
        %sscale_177 = memref.load %in_v[%ssidx_177] : memref<1511xi32>
        %dilation_177 = arith.constant 2 : i32
        %padding_177 = arith.constant 8 : i32
        %ki0_177 = arith.constant 0 : i32
        %ki1_177 = arith.constant 2 : i32
        %ki2_177 = arith.constant 5 : i32
        %pad1_177 = arith.constant 0 : i32
        %outidx_177 = arith.constant 177 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_177, %smean_177, %sscale_177, %dilation_177, %padding_177, %ki0_177, %ki1_177, %ki2_177, %pad1_177, %out_v, %outidx_177) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 178: dilation=2 padding=8 kernel=(0,2,6) padding1=1
        %fidx_178 = arith.constant 178 : index
        %bidx_178 = arith.constant 429 : index
        %smidx_178 = arith.constant 849 : index
        %ssidx_178 = arith.constant 1269 : index
        %bias_178 = memref.load %in_v[%bidx_178] : memref<1511xi32>
        %smean_178 = memref.load %in_v[%smidx_178] : memref<1511xi32>
        %sscale_178 = memref.load %in_v[%ssidx_178] : memref<1511xi32>
        %dilation_178 = arith.constant 2 : i32
        %padding_178 = arith.constant 8 : i32
        %ki0_178 = arith.constant 0 : i32
        %ki1_178 = arith.constant 2 : i32
        %ki2_178 = arith.constant 6 : i32
        %pad1_178 = arith.constant 1 : i32
        %outidx_178 = arith.constant 178 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_178, %smean_178, %sscale_178, %dilation_178, %padding_178, %ki0_178, %ki1_178, %ki2_178, %pad1_178, %out_v, %outidx_178) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 179: dilation=2 padding=8 kernel=(0,2,7) padding1=0
        %fidx_179 = arith.constant 179 : index
        %bidx_179 = arith.constant 430 : index
        %smidx_179 = arith.constant 850 : index
        %ssidx_179 = arith.constant 1270 : index
        %bias_179 = memref.load %in_v[%bidx_179] : memref<1511xi32>
        %smean_179 = memref.load %in_v[%smidx_179] : memref<1511xi32>
        %sscale_179 = memref.load %in_v[%ssidx_179] : memref<1511xi32>
        %dilation_179 = arith.constant 2 : i32
        %padding_179 = arith.constant 8 : i32
        %ki0_179 = arith.constant 0 : i32
        %ki1_179 = arith.constant 2 : i32
        %ki2_179 = arith.constant 7 : i32
        %pad1_179 = arith.constant 0 : i32
        %outidx_179 = arith.constant 179 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_179, %smean_179, %sscale_179, %dilation_179, %padding_179, %ki0_179, %ki1_179, %ki2_179, %pad1_179, %out_v, %outidx_179) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 180: dilation=2 padding=8 kernel=(0,2,8) padding1=1
        %fidx_180 = arith.constant 180 : index
        %bidx_180 = arith.constant 431 : index
        %smidx_180 = arith.constant 851 : index
        %ssidx_180 = arith.constant 1271 : index
        %bias_180 = memref.load %in_v[%bidx_180] : memref<1511xi32>
        %smean_180 = memref.load %in_v[%smidx_180] : memref<1511xi32>
        %sscale_180 = memref.load %in_v[%ssidx_180] : memref<1511xi32>
        %dilation_180 = arith.constant 2 : i32
        %padding_180 = arith.constant 8 : i32
        %ki0_180 = arith.constant 0 : i32
        %ki1_180 = arith.constant 2 : i32
        %ki2_180 = arith.constant 8 : i32
        %pad1_180 = arith.constant 1 : i32
        %outidx_180 = arith.constant 180 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_180, %smean_180, %sscale_180, %dilation_180, %padding_180, %ki0_180, %ki1_180, %ki2_180, %pad1_180, %out_v, %outidx_180) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 181: dilation=2 padding=8 kernel=(0,3,4) padding1=0
        %fidx_181 = arith.constant 181 : index
        %bidx_181 = arith.constant 432 : index
        %smidx_181 = arith.constant 852 : index
        %ssidx_181 = arith.constant 1272 : index
        %bias_181 = memref.load %in_v[%bidx_181] : memref<1511xi32>
        %smean_181 = memref.load %in_v[%smidx_181] : memref<1511xi32>
        %sscale_181 = memref.load %in_v[%ssidx_181] : memref<1511xi32>
        %dilation_181 = arith.constant 2 : i32
        %padding_181 = arith.constant 8 : i32
        %ki0_181 = arith.constant 0 : i32
        %ki1_181 = arith.constant 3 : i32
        %ki2_181 = arith.constant 4 : i32
        %pad1_181 = arith.constant 0 : i32
        %outidx_181 = arith.constant 181 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_181, %smean_181, %sscale_181, %dilation_181, %padding_181, %ki0_181, %ki1_181, %ki2_181, %pad1_181, %out_v, %outidx_181) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 182: dilation=2 padding=8 kernel=(0,3,5) padding1=1
        %fidx_182 = arith.constant 182 : index
        %bidx_182 = arith.constant 433 : index
        %smidx_182 = arith.constant 853 : index
        %ssidx_182 = arith.constant 1273 : index
        %bias_182 = memref.load %in_v[%bidx_182] : memref<1511xi32>
        %smean_182 = memref.load %in_v[%smidx_182] : memref<1511xi32>
        %sscale_182 = memref.load %in_v[%ssidx_182] : memref<1511xi32>
        %dilation_182 = arith.constant 2 : i32
        %padding_182 = arith.constant 8 : i32
        %ki0_182 = arith.constant 0 : i32
        %ki1_182 = arith.constant 3 : i32
        %ki2_182 = arith.constant 5 : i32
        %pad1_182 = arith.constant 1 : i32
        %outidx_182 = arith.constant 182 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_182, %smean_182, %sscale_182, %dilation_182, %padding_182, %ki0_182, %ki1_182, %ki2_182, %pad1_182, %out_v, %outidx_182) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 183: dilation=2 padding=8 kernel=(0,3,6) padding1=0
        %fidx_183 = arith.constant 183 : index
        %bidx_183 = arith.constant 434 : index
        %smidx_183 = arith.constant 854 : index
        %ssidx_183 = arith.constant 1274 : index
        %bias_183 = memref.load %in_v[%bidx_183] : memref<1511xi32>
        %smean_183 = memref.load %in_v[%smidx_183] : memref<1511xi32>
        %sscale_183 = memref.load %in_v[%ssidx_183] : memref<1511xi32>
        %dilation_183 = arith.constant 2 : i32
        %padding_183 = arith.constant 8 : i32
        %ki0_183 = arith.constant 0 : i32
        %ki1_183 = arith.constant 3 : i32
        %ki2_183 = arith.constant 6 : i32
        %pad1_183 = arith.constant 0 : i32
        %outidx_183 = arith.constant 183 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_183, %smean_183, %sscale_183, %dilation_183, %padding_183, %ki0_183, %ki1_183, %ki2_183, %pad1_183, %out_v, %outidx_183) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 184: dilation=2 padding=8 kernel=(0,3,7) padding1=1
        %fidx_184 = arith.constant 184 : index
        %bidx_184 = arith.constant 435 : index
        %smidx_184 = arith.constant 855 : index
        %ssidx_184 = arith.constant 1275 : index
        %bias_184 = memref.load %in_v[%bidx_184] : memref<1511xi32>
        %smean_184 = memref.load %in_v[%smidx_184] : memref<1511xi32>
        %sscale_184 = memref.load %in_v[%ssidx_184] : memref<1511xi32>
        %dilation_184 = arith.constant 2 : i32
        %padding_184 = arith.constant 8 : i32
        %ki0_184 = arith.constant 0 : i32
        %ki1_184 = arith.constant 3 : i32
        %ki2_184 = arith.constant 7 : i32
        %pad1_184 = arith.constant 1 : i32
        %outidx_184 = arith.constant 184 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_184, %smean_184, %sscale_184, %dilation_184, %padding_184, %ki0_184, %ki1_184, %ki2_184, %pad1_184, %out_v, %outidx_184) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 185: dilation=2 padding=8 kernel=(0,3,8) padding1=0
        %fidx_185 = arith.constant 185 : index
        %bidx_185 = arith.constant 436 : index
        %smidx_185 = arith.constant 856 : index
        %ssidx_185 = arith.constant 1276 : index
        %bias_185 = memref.load %in_v[%bidx_185] : memref<1511xi32>
        %smean_185 = memref.load %in_v[%smidx_185] : memref<1511xi32>
        %sscale_185 = memref.load %in_v[%ssidx_185] : memref<1511xi32>
        %dilation_185 = arith.constant 2 : i32
        %padding_185 = arith.constant 8 : i32
        %ki0_185 = arith.constant 0 : i32
        %ki1_185 = arith.constant 3 : i32
        %ki2_185 = arith.constant 8 : i32
        %pad1_185 = arith.constant 0 : i32
        %outidx_185 = arith.constant 185 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_185, %smean_185, %sscale_185, %dilation_185, %padding_185, %ki0_185, %ki1_185, %ki2_185, %pad1_185, %out_v, %outidx_185) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 186: dilation=2 padding=8 kernel=(0,4,5) padding1=1
        %fidx_186 = arith.constant 186 : index
        %bidx_186 = arith.constant 437 : index
        %smidx_186 = arith.constant 857 : index
        %ssidx_186 = arith.constant 1277 : index
        %bias_186 = memref.load %in_v[%bidx_186] : memref<1511xi32>
        %smean_186 = memref.load %in_v[%smidx_186] : memref<1511xi32>
        %sscale_186 = memref.load %in_v[%ssidx_186] : memref<1511xi32>
        %dilation_186 = arith.constant 2 : i32
        %padding_186 = arith.constant 8 : i32
        %ki0_186 = arith.constant 0 : i32
        %ki1_186 = arith.constant 4 : i32
        %ki2_186 = arith.constant 5 : i32
        %pad1_186 = arith.constant 1 : i32
        %outidx_186 = arith.constant 186 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_186, %smean_186, %sscale_186, %dilation_186, %padding_186, %ki0_186, %ki1_186, %ki2_186, %pad1_186, %out_v, %outidx_186) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 187: dilation=2 padding=8 kernel=(0,4,6) padding1=0
        %fidx_187 = arith.constant 187 : index
        %bidx_187 = arith.constant 438 : index
        %smidx_187 = arith.constant 858 : index
        %ssidx_187 = arith.constant 1278 : index
        %bias_187 = memref.load %in_v[%bidx_187] : memref<1511xi32>
        %smean_187 = memref.load %in_v[%smidx_187] : memref<1511xi32>
        %sscale_187 = memref.load %in_v[%ssidx_187] : memref<1511xi32>
        %dilation_187 = arith.constant 2 : i32
        %padding_187 = arith.constant 8 : i32
        %ki0_187 = arith.constant 0 : i32
        %ki1_187 = arith.constant 4 : i32
        %ki2_187 = arith.constant 6 : i32
        %pad1_187 = arith.constant 0 : i32
        %outidx_187 = arith.constant 187 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_187, %smean_187, %sscale_187, %dilation_187, %padding_187, %ki0_187, %ki1_187, %ki2_187, %pad1_187, %out_v, %outidx_187) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 188: dilation=2 padding=8 kernel=(0,4,7) padding1=1
        %fidx_188 = arith.constant 188 : index
        %bidx_188 = arith.constant 439 : index
        %smidx_188 = arith.constant 859 : index
        %ssidx_188 = arith.constant 1279 : index
        %bias_188 = memref.load %in_v[%bidx_188] : memref<1511xi32>
        %smean_188 = memref.load %in_v[%smidx_188] : memref<1511xi32>
        %sscale_188 = memref.load %in_v[%ssidx_188] : memref<1511xi32>
        %dilation_188 = arith.constant 2 : i32
        %padding_188 = arith.constant 8 : i32
        %ki0_188 = arith.constant 0 : i32
        %ki1_188 = arith.constant 4 : i32
        %ki2_188 = arith.constant 7 : i32
        %pad1_188 = arith.constant 1 : i32
        %outidx_188 = arith.constant 188 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_188, %smean_188, %sscale_188, %dilation_188, %padding_188, %ki0_188, %ki1_188, %ki2_188, %pad1_188, %out_v, %outidx_188) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 189: dilation=2 padding=8 kernel=(0,4,8) padding1=0
        %fidx_189 = arith.constant 189 : index
        %bidx_189 = arith.constant 440 : index
        %smidx_189 = arith.constant 860 : index
        %ssidx_189 = arith.constant 1280 : index
        %bias_189 = memref.load %in_v[%bidx_189] : memref<1511xi32>
        %smean_189 = memref.load %in_v[%smidx_189] : memref<1511xi32>
        %sscale_189 = memref.load %in_v[%ssidx_189] : memref<1511xi32>
        %dilation_189 = arith.constant 2 : i32
        %padding_189 = arith.constant 8 : i32
        %ki0_189 = arith.constant 0 : i32
        %ki1_189 = arith.constant 4 : i32
        %ki2_189 = arith.constant 8 : i32
        %pad1_189 = arith.constant 0 : i32
        %outidx_189 = arith.constant 189 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_189, %smean_189, %sscale_189, %dilation_189, %padding_189, %ki0_189, %ki1_189, %ki2_189, %pad1_189, %out_v, %outidx_189) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 190: dilation=2 padding=8 kernel=(0,5,6) padding1=1
        %fidx_190 = arith.constant 190 : index
        %bidx_190 = arith.constant 441 : index
        %smidx_190 = arith.constant 861 : index
        %ssidx_190 = arith.constant 1281 : index
        %bias_190 = memref.load %in_v[%bidx_190] : memref<1511xi32>
        %smean_190 = memref.load %in_v[%smidx_190] : memref<1511xi32>
        %sscale_190 = memref.load %in_v[%ssidx_190] : memref<1511xi32>
        %dilation_190 = arith.constant 2 : i32
        %padding_190 = arith.constant 8 : i32
        %ki0_190 = arith.constant 0 : i32
        %ki1_190 = arith.constant 5 : i32
        %ki2_190 = arith.constant 6 : i32
        %pad1_190 = arith.constant 1 : i32
        %outidx_190 = arith.constant 190 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_190, %smean_190, %sscale_190, %dilation_190, %padding_190, %ki0_190, %ki1_190, %ki2_190, %pad1_190, %out_v, %outidx_190) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 191: dilation=2 padding=8 kernel=(0,5,7) padding1=0
        %fidx_191 = arith.constant 191 : index
        %bidx_191 = arith.constant 442 : index
        %smidx_191 = arith.constant 862 : index
        %ssidx_191 = arith.constant 1282 : index
        %bias_191 = memref.load %in_v[%bidx_191] : memref<1511xi32>
        %smean_191 = memref.load %in_v[%smidx_191] : memref<1511xi32>
        %sscale_191 = memref.load %in_v[%ssidx_191] : memref<1511xi32>
        %dilation_191 = arith.constant 2 : i32
        %padding_191 = arith.constant 8 : i32
        %ki0_191 = arith.constant 0 : i32
        %ki1_191 = arith.constant 5 : i32
        %ki2_191 = arith.constant 7 : i32
        %pad1_191 = arith.constant 0 : i32
        %outidx_191 = arith.constant 191 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_191, %smean_191, %sscale_191, %dilation_191, %padding_191, %ki0_191, %ki1_191, %ki2_191, %pad1_191, %out_v, %outidx_191) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 192: dilation=2 padding=8 kernel=(0,5,8) padding1=1
        %fidx_192 = arith.constant 192 : index
        %bidx_192 = arith.constant 443 : index
        %smidx_192 = arith.constant 863 : index
        %ssidx_192 = arith.constant 1283 : index
        %bias_192 = memref.load %in_v[%bidx_192] : memref<1511xi32>
        %smean_192 = memref.load %in_v[%smidx_192] : memref<1511xi32>
        %sscale_192 = memref.load %in_v[%ssidx_192] : memref<1511xi32>
        %dilation_192 = arith.constant 2 : i32
        %padding_192 = arith.constant 8 : i32
        %ki0_192 = arith.constant 0 : i32
        %ki1_192 = arith.constant 5 : i32
        %ki2_192 = arith.constant 8 : i32
        %pad1_192 = arith.constant 1 : i32
        %outidx_192 = arith.constant 192 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_192, %smean_192, %sscale_192, %dilation_192, %padding_192, %ki0_192, %ki1_192, %ki2_192, %pad1_192, %out_v, %outidx_192) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 193: dilation=2 padding=8 kernel=(0,6,7) padding1=0
        %fidx_193 = arith.constant 193 : index
        %bidx_193 = arith.constant 444 : index
        %smidx_193 = arith.constant 864 : index
        %ssidx_193 = arith.constant 1284 : index
        %bias_193 = memref.load %in_v[%bidx_193] : memref<1511xi32>
        %smean_193 = memref.load %in_v[%smidx_193] : memref<1511xi32>
        %sscale_193 = memref.load %in_v[%ssidx_193] : memref<1511xi32>
        %dilation_193 = arith.constant 2 : i32
        %padding_193 = arith.constant 8 : i32
        %ki0_193 = arith.constant 0 : i32
        %ki1_193 = arith.constant 6 : i32
        %ki2_193 = arith.constant 7 : i32
        %pad1_193 = arith.constant 0 : i32
        %outidx_193 = arith.constant 193 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_193, %smean_193, %sscale_193, %dilation_193, %padding_193, %ki0_193, %ki1_193, %ki2_193, %pad1_193, %out_v, %outidx_193) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 194: dilation=2 padding=8 kernel=(0,6,8) padding1=1
        %fidx_194 = arith.constant 194 : index
        %bidx_194 = arith.constant 445 : index
        %smidx_194 = arith.constant 865 : index
        %ssidx_194 = arith.constant 1285 : index
        %bias_194 = memref.load %in_v[%bidx_194] : memref<1511xi32>
        %smean_194 = memref.load %in_v[%smidx_194] : memref<1511xi32>
        %sscale_194 = memref.load %in_v[%ssidx_194] : memref<1511xi32>
        %dilation_194 = arith.constant 2 : i32
        %padding_194 = arith.constant 8 : i32
        %ki0_194 = arith.constant 0 : i32
        %ki1_194 = arith.constant 6 : i32
        %ki2_194 = arith.constant 8 : i32
        %pad1_194 = arith.constant 1 : i32
        %outidx_194 = arith.constant 194 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_194, %smean_194, %sscale_194, %dilation_194, %padding_194, %ki0_194, %ki1_194, %ki2_194, %pad1_194, %out_v, %outidx_194) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 195: dilation=2 padding=8 kernel=(0,7,8) padding1=0
        %fidx_195 = arith.constant 195 : index
        %bidx_195 = arith.constant 446 : index
        %smidx_195 = arith.constant 866 : index
        %ssidx_195 = arith.constant 1286 : index
        %bias_195 = memref.load %in_v[%bidx_195] : memref<1511xi32>
        %smean_195 = memref.load %in_v[%smidx_195] : memref<1511xi32>
        %sscale_195 = memref.load %in_v[%ssidx_195] : memref<1511xi32>
        %dilation_195 = arith.constant 2 : i32
        %padding_195 = arith.constant 8 : i32
        %ki0_195 = arith.constant 0 : i32
        %ki1_195 = arith.constant 7 : i32
        %ki2_195 = arith.constant 8 : i32
        %pad1_195 = arith.constant 0 : i32
        %outidx_195 = arith.constant 195 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_195, %smean_195, %sscale_195, %dilation_195, %padding_195, %ki0_195, %ki1_195, %ki2_195, %pad1_195, %out_v, %outidx_195) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 196: dilation=2 padding=8 kernel=(1,2,3) padding1=1
        %fidx_196 = arith.constant 196 : index
        %bidx_196 = arith.constant 447 : index
        %smidx_196 = arith.constant 867 : index
        %ssidx_196 = arith.constant 1287 : index
        %bias_196 = memref.load %in_v[%bidx_196] : memref<1511xi32>
        %smean_196 = memref.load %in_v[%smidx_196] : memref<1511xi32>
        %sscale_196 = memref.load %in_v[%ssidx_196] : memref<1511xi32>
        %dilation_196 = arith.constant 2 : i32
        %padding_196 = arith.constant 8 : i32
        %ki0_196 = arith.constant 1 : i32
        %ki1_196 = arith.constant 2 : i32
        %ki2_196 = arith.constant 3 : i32
        %pad1_196 = arith.constant 1 : i32
        %outidx_196 = arith.constant 196 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_196, %smean_196, %sscale_196, %dilation_196, %padding_196, %ki0_196, %ki1_196, %ki2_196, %pad1_196, %out_v, %outidx_196) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 197: dilation=2 padding=8 kernel=(1,2,4) padding1=0
        %fidx_197 = arith.constant 197 : index
        %bidx_197 = arith.constant 448 : index
        %smidx_197 = arith.constant 868 : index
        %ssidx_197 = arith.constant 1288 : index
        %bias_197 = memref.load %in_v[%bidx_197] : memref<1511xi32>
        %smean_197 = memref.load %in_v[%smidx_197] : memref<1511xi32>
        %sscale_197 = memref.load %in_v[%ssidx_197] : memref<1511xi32>
        %dilation_197 = arith.constant 2 : i32
        %padding_197 = arith.constant 8 : i32
        %ki0_197 = arith.constant 1 : i32
        %ki1_197 = arith.constant 2 : i32
        %ki2_197 = arith.constant 4 : i32
        %pad1_197 = arith.constant 0 : i32
        %outidx_197 = arith.constant 197 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_197, %smean_197, %sscale_197, %dilation_197, %padding_197, %ki0_197, %ki1_197, %ki2_197, %pad1_197, %out_v, %outidx_197) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 198: dilation=2 padding=8 kernel=(1,2,5) padding1=1
        %fidx_198 = arith.constant 198 : index
        %bidx_198 = arith.constant 449 : index
        %smidx_198 = arith.constant 869 : index
        %ssidx_198 = arith.constant 1289 : index
        %bias_198 = memref.load %in_v[%bidx_198] : memref<1511xi32>
        %smean_198 = memref.load %in_v[%smidx_198] : memref<1511xi32>
        %sscale_198 = memref.load %in_v[%ssidx_198] : memref<1511xi32>
        %dilation_198 = arith.constant 2 : i32
        %padding_198 = arith.constant 8 : i32
        %ki0_198 = arith.constant 1 : i32
        %ki1_198 = arith.constant 2 : i32
        %ki2_198 = arith.constant 5 : i32
        %pad1_198 = arith.constant 1 : i32
        %outidx_198 = arith.constant 198 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_198, %smean_198, %sscale_198, %dilation_198, %padding_198, %ki0_198, %ki1_198, %ki2_198, %pad1_198, %out_v, %outidx_198) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 199: dilation=2 padding=8 kernel=(1,2,6) padding1=0
        %fidx_199 = arith.constant 199 : index
        %bidx_199 = arith.constant 450 : index
        %smidx_199 = arith.constant 870 : index
        %ssidx_199 = arith.constant 1290 : index
        %bias_199 = memref.load %in_v[%bidx_199] : memref<1511xi32>
        %smean_199 = memref.load %in_v[%smidx_199] : memref<1511xi32>
        %sscale_199 = memref.load %in_v[%ssidx_199] : memref<1511xi32>
        %dilation_199 = arith.constant 2 : i32
        %padding_199 = arith.constant 8 : i32
        %ki0_199 = arith.constant 1 : i32
        %ki1_199 = arith.constant 2 : i32
        %ki2_199 = arith.constant 6 : i32
        %pad1_199 = arith.constant 0 : i32
        %outidx_199 = arith.constant 199 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_199, %smean_199, %sscale_199, %dilation_199, %padding_199, %ki0_199, %ki1_199, %ki2_199, %pad1_199, %out_v, %outidx_199) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 200: dilation=2 padding=8 kernel=(1,2,7) padding1=1
        %fidx_200 = arith.constant 200 : index
        %bidx_200 = arith.constant 451 : index
        %smidx_200 = arith.constant 871 : index
        %ssidx_200 = arith.constant 1291 : index
        %bias_200 = memref.load %in_v[%bidx_200] : memref<1511xi32>
        %smean_200 = memref.load %in_v[%smidx_200] : memref<1511xi32>
        %sscale_200 = memref.load %in_v[%ssidx_200] : memref<1511xi32>
        %dilation_200 = arith.constant 2 : i32
        %padding_200 = arith.constant 8 : i32
        %ki0_200 = arith.constant 1 : i32
        %ki1_200 = arith.constant 2 : i32
        %ki2_200 = arith.constant 7 : i32
        %pad1_200 = arith.constant 1 : i32
        %outidx_200 = arith.constant 200 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_200, %smean_200, %sscale_200, %dilation_200, %padding_200, %ki0_200, %ki1_200, %ki2_200, %pad1_200, %out_v, %outidx_200) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 201: dilation=2 padding=8 kernel=(1,2,8) padding1=0
        %fidx_201 = arith.constant 201 : index
        %bidx_201 = arith.constant 452 : index
        %smidx_201 = arith.constant 872 : index
        %ssidx_201 = arith.constant 1292 : index
        %bias_201 = memref.load %in_v[%bidx_201] : memref<1511xi32>
        %smean_201 = memref.load %in_v[%smidx_201] : memref<1511xi32>
        %sscale_201 = memref.load %in_v[%ssidx_201] : memref<1511xi32>
        %dilation_201 = arith.constant 2 : i32
        %padding_201 = arith.constant 8 : i32
        %ki0_201 = arith.constant 1 : i32
        %ki1_201 = arith.constant 2 : i32
        %ki2_201 = arith.constant 8 : i32
        %pad1_201 = arith.constant 0 : i32
        %outidx_201 = arith.constant 201 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_201, %smean_201, %sscale_201, %dilation_201, %padding_201, %ki0_201, %ki1_201, %ki2_201, %pad1_201, %out_v, %outidx_201) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 202: dilation=2 padding=8 kernel=(1,3,4) padding1=1
        %fidx_202 = arith.constant 202 : index
        %bidx_202 = arith.constant 453 : index
        %smidx_202 = arith.constant 873 : index
        %ssidx_202 = arith.constant 1293 : index
        %bias_202 = memref.load %in_v[%bidx_202] : memref<1511xi32>
        %smean_202 = memref.load %in_v[%smidx_202] : memref<1511xi32>
        %sscale_202 = memref.load %in_v[%ssidx_202] : memref<1511xi32>
        %dilation_202 = arith.constant 2 : i32
        %padding_202 = arith.constant 8 : i32
        %ki0_202 = arith.constant 1 : i32
        %ki1_202 = arith.constant 3 : i32
        %ki2_202 = arith.constant 4 : i32
        %pad1_202 = arith.constant 1 : i32
        %outidx_202 = arith.constant 202 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_202, %smean_202, %sscale_202, %dilation_202, %padding_202, %ki0_202, %ki1_202, %ki2_202, %pad1_202, %out_v, %outidx_202) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 203: dilation=2 padding=8 kernel=(1,3,5) padding1=0
        %fidx_203 = arith.constant 203 : index
        %bidx_203 = arith.constant 454 : index
        %smidx_203 = arith.constant 874 : index
        %ssidx_203 = arith.constant 1294 : index
        %bias_203 = memref.load %in_v[%bidx_203] : memref<1511xi32>
        %smean_203 = memref.load %in_v[%smidx_203] : memref<1511xi32>
        %sscale_203 = memref.load %in_v[%ssidx_203] : memref<1511xi32>
        %dilation_203 = arith.constant 2 : i32
        %padding_203 = arith.constant 8 : i32
        %ki0_203 = arith.constant 1 : i32
        %ki1_203 = arith.constant 3 : i32
        %ki2_203 = arith.constant 5 : i32
        %pad1_203 = arith.constant 0 : i32
        %outidx_203 = arith.constant 203 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_203, %smean_203, %sscale_203, %dilation_203, %padding_203, %ki0_203, %ki1_203, %ki2_203, %pad1_203, %out_v, %outidx_203) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 204: dilation=2 padding=8 kernel=(1,3,6) padding1=1
        %fidx_204 = arith.constant 204 : index
        %bidx_204 = arith.constant 455 : index
        %smidx_204 = arith.constant 875 : index
        %ssidx_204 = arith.constant 1295 : index
        %bias_204 = memref.load %in_v[%bidx_204] : memref<1511xi32>
        %smean_204 = memref.load %in_v[%smidx_204] : memref<1511xi32>
        %sscale_204 = memref.load %in_v[%ssidx_204] : memref<1511xi32>
        %dilation_204 = arith.constant 2 : i32
        %padding_204 = arith.constant 8 : i32
        %ki0_204 = arith.constant 1 : i32
        %ki1_204 = arith.constant 3 : i32
        %ki2_204 = arith.constant 6 : i32
        %pad1_204 = arith.constant 1 : i32
        %outidx_204 = arith.constant 204 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_204, %smean_204, %sscale_204, %dilation_204, %padding_204, %ki0_204, %ki1_204, %ki2_204, %pad1_204, %out_v, %outidx_204) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 205: dilation=2 padding=8 kernel=(1,3,7) padding1=0
        %fidx_205 = arith.constant 205 : index
        %bidx_205 = arith.constant 456 : index
        %smidx_205 = arith.constant 876 : index
        %ssidx_205 = arith.constant 1296 : index
        %bias_205 = memref.load %in_v[%bidx_205] : memref<1511xi32>
        %smean_205 = memref.load %in_v[%smidx_205] : memref<1511xi32>
        %sscale_205 = memref.load %in_v[%ssidx_205] : memref<1511xi32>
        %dilation_205 = arith.constant 2 : i32
        %padding_205 = arith.constant 8 : i32
        %ki0_205 = arith.constant 1 : i32
        %ki1_205 = arith.constant 3 : i32
        %ki2_205 = arith.constant 7 : i32
        %pad1_205 = arith.constant 0 : i32
        %outidx_205 = arith.constant 205 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_205, %smean_205, %sscale_205, %dilation_205, %padding_205, %ki0_205, %ki1_205, %ki2_205, %pad1_205, %out_v, %outidx_205) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 206: dilation=2 padding=8 kernel=(1,3,8) padding1=1
        %fidx_206 = arith.constant 206 : index
        %bidx_206 = arith.constant 457 : index
        %smidx_206 = arith.constant 877 : index
        %ssidx_206 = arith.constant 1297 : index
        %bias_206 = memref.load %in_v[%bidx_206] : memref<1511xi32>
        %smean_206 = memref.load %in_v[%smidx_206] : memref<1511xi32>
        %sscale_206 = memref.load %in_v[%ssidx_206] : memref<1511xi32>
        %dilation_206 = arith.constant 2 : i32
        %padding_206 = arith.constant 8 : i32
        %ki0_206 = arith.constant 1 : i32
        %ki1_206 = arith.constant 3 : i32
        %ki2_206 = arith.constant 8 : i32
        %pad1_206 = arith.constant 1 : i32
        %outidx_206 = arith.constant 206 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_206, %smean_206, %sscale_206, %dilation_206, %padding_206, %ki0_206, %ki1_206, %ki2_206, %pad1_206, %out_v, %outidx_206) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 207: dilation=2 padding=8 kernel=(1,4,5) padding1=0
        %fidx_207 = arith.constant 207 : index
        %bidx_207 = arith.constant 458 : index
        %smidx_207 = arith.constant 878 : index
        %ssidx_207 = arith.constant 1298 : index
        %bias_207 = memref.load %in_v[%bidx_207] : memref<1511xi32>
        %smean_207 = memref.load %in_v[%smidx_207] : memref<1511xi32>
        %sscale_207 = memref.load %in_v[%ssidx_207] : memref<1511xi32>
        %dilation_207 = arith.constant 2 : i32
        %padding_207 = arith.constant 8 : i32
        %ki0_207 = arith.constant 1 : i32
        %ki1_207 = arith.constant 4 : i32
        %ki2_207 = arith.constant 5 : i32
        %pad1_207 = arith.constant 0 : i32
        %outidx_207 = arith.constant 207 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_207, %smean_207, %sscale_207, %dilation_207, %padding_207, %ki0_207, %ki1_207, %ki2_207, %pad1_207, %out_v, %outidx_207) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 208: dilation=2 padding=8 kernel=(1,4,6) padding1=1
        %fidx_208 = arith.constant 208 : index
        %bidx_208 = arith.constant 459 : index
        %smidx_208 = arith.constant 879 : index
        %ssidx_208 = arith.constant 1299 : index
        %bias_208 = memref.load %in_v[%bidx_208] : memref<1511xi32>
        %smean_208 = memref.load %in_v[%smidx_208] : memref<1511xi32>
        %sscale_208 = memref.load %in_v[%ssidx_208] : memref<1511xi32>
        %dilation_208 = arith.constant 2 : i32
        %padding_208 = arith.constant 8 : i32
        %ki0_208 = arith.constant 1 : i32
        %ki1_208 = arith.constant 4 : i32
        %ki2_208 = arith.constant 6 : i32
        %pad1_208 = arith.constant 1 : i32
        %outidx_208 = arith.constant 208 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_208, %smean_208, %sscale_208, %dilation_208, %padding_208, %ki0_208, %ki1_208, %ki2_208, %pad1_208, %out_v, %outidx_208) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 209: dilation=2 padding=8 kernel=(1,4,7) padding1=0
        %fidx_209 = arith.constant 209 : index
        %bidx_209 = arith.constant 460 : index
        %smidx_209 = arith.constant 880 : index
        %ssidx_209 = arith.constant 1300 : index
        %bias_209 = memref.load %in_v[%bidx_209] : memref<1511xi32>
        %smean_209 = memref.load %in_v[%smidx_209] : memref<1511xi32>
        %sscale_209 = memref.load %in_v[%ssidx_209] : memref<1511xi32>
        %dilation_209 = arith.constant 2 : i32
        %padding_209 = arith.constant 8 : i32
        %ki0_209 = arith.constant 1 : i32
        %ki1_209 = arith.constant 4 : i32
        %ki2_209 = arith.constant 7 : i32
        %pad1_209 = arith.constant 0 : i32
        %outidx_209 = arith.constant 209 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_209, %smean_209, %sscale_209, %dilation_209, %padding_209, %ki0_209, %ki1_209, %ki2_209, %pad1_209, %out_v, %outidx_209) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 210: dilation=2 padding=8 kernel=(1,4,8) padding1=1
        %fidx_210 = arith.constant 210 : index
        %bidx_210 = arith.constant 461 : index
        %smidx_210 = arith.constant 881 : index
        %ssidx_210 = arith.constant 1301 : index
        %bias_210 = memref.load %in_v[%bidx_210] : memref<1511xi32>
        %smean_210 = memref.load %in_v[%smidx_210] : memref<1511xi32>
        %sscale_210 = memref.load %in_v[%ssidx_210] : memref<1511xi32>
        %dilation_210 = arith.constant 2 : i32
        %padding_210 = arith.constant 8 : i32
        %ki0_210 = arith.constant 1 : i32
        %ki1_210 = arith.constant 4 : i32
        %ki2_210 = arith.constant 8 : i32
        %pad1_210 = arith.constant 1 : i32
        %outidx_210 = arith.constant 210 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_210, %smean_210, %sscale_210, %dilation_210, %padding_210, %ki0_210, %ki1_210, %ki2_210, %pad1_210, %out_v, %outidx_210) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 211: dilation=2 padding=8 kernel=(1,5,6) padding1=0
        %fidx_211 = arith.constant 211 : index
        %bidx_211 = arith.constant 462 : index
        %smidx_211 = arith.constant 882 : index
        %ssidx_211 = arith.constant 1302 : index
        %bias_211 = memref.load %in_v[%bidx_211] : memref<1511xi32>
        %smean_211 = memref.load %in_v[%smidx_211] : memref<1511xi32>
        %sscale_211 = memref.load %in_v[%ssidx_211] : memref<1511xi32>
        %dilation_211 = arith.constant 2 : i32
        %padding_211 = arith.constant 8 : i32
        %ki0_211 = arith.constant 1 : i32
        %ki1_211 = arith.constant 5 : i32
        %ki2_211 = arith.constant 6 : i32
        %pad1_211 = arith.constant 0 : i32
        %outidx_211 = arith.constant 211 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_211, %smean_211, %sscale_211, %dilation_211, %padding_211, %ki0_211, %ki1_211, %ki2_211, %pad1_211, %out_v, %outidx_211) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 212: dilation=2 padding=8 kernel=(1,5,7) padding1=1
        %fidx_212 = arith.constant 212 : index
        %bidx_212 = arith.constant 463 : index
        %smidx_212 = arith.constant 883 : index
        %ssidx_212 = arith.constant 1303 : index
        %bias_212 = memref.load %in_v[%bidx_212] : memref<1511xi32>
        %smean_212 = memref.load %in_v[%smidx_212] : memref<1511xi32>
        %sscale_212 = memref.load %in_v[%ssidx_212] : memref<1511xi32>
        %dilation_212 = arith.constant 2 : i32
        %padding_212 = arith.constant 8 : i32
        %ki0_212 = arith.constant 1 : i32
        %ki1_212 = arith.constant 5 : i32
        %ki2_212 = arith.constant 7 : i32
        %pad1_212 = arith.constant 1 : i32
        %outidx_212 = arith.constant 212 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_212, %smean_212, %sscale_212, %dilation_212, %padding_212, %ki0_212, %ki1_212, %ki2_212, %pad1_212, %out_v, %outidx_212) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 213: dilation=2 padding=8 kernel=(1,5,8) padding1=0
        %fidx_213 = arith.constant 213 : index
        %bidx_213 = arith.constant 464 : index
        %smidx_213 = arith.constant 884 : index
        %ssidx_213 = arith.constant 1304 : index
        %bias_213 = memref.load %in_v[%bidx_213] : memref<1511xi32>
        %smean_213 = memref.load %in_v[%smidx_213] : memref<1511xi32>
        %sscale_213 = memref.load %in_v[%ssidx_213] : memref<1511xi32>
        %dilation_213 = arith.constant 2 : i32
        %padding_213 = arith.constant 8 : i32
        %ki0_213 = arith.constant 1 : i32
        %ki1_213 = arith.constant 5 : i32
        %ki2_213 = arith.constant 8 : i32
        %pad1_213 = arith.constant 0 : i32
        %outidx_213 = arith.constant 213 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_213, %smean_213, %sscale_213, %dilation_213, %padding_213, %ki0_213, %ki1_213, %ki2_213, %pad1_213, %out_v, %outidx_213) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 214: dilation=2 padding=8 kernel=(1,6,7) padding1=1
        %fidx_214 = arith.constant 214 : index
        %bidx_214 = arith.constant 465 : index
        %smidx_214 = arith.constant 885 : index
        %ssidx_214 = arith.constant 1305 : index
        %bias_214 = memref.load %in_v[%bidx_214] : memref<1511xi32>
        %smean_214 = memref.load %in_v[%smidx_214] : memref<1511xi32>
        %sscale_214 = memref.load %in_v[%ssidx_214] : memref<1511xi32>
        %dilation_214 = arith.constant 2 : i32
        %padding_214 = arith.constant 8 : i32
        %ki0_214 = arith.constant 1 : i32
        %ki1_214 = arith.constant 6 : i32
        %ki2_214 = arith.constant 7 : i32
        %pad1_214 = arith.constant 1 : i32
        %outidx_214 = arith.constant 214 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_214, %smean_214, %sscale_214, %dilation_214, %padding_214, %ki0_214, %ki1_214, %ki2_214, %pad1_214, %out_v, %outidx_214) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 215: dilation=2 padding=8 kernel=(1,6,8) padding1=0
        %fidx_215 = arith.constant 215 : index
        %bidx_215 = arith.constant 466 : index
        %smidx_215 = arith.constant 886 : index
        %ssidx_215 = arith.constant 1306 : index
        %bias_215 = memref.load %in_v[%bidx_215] : memref<1511xi32>
        %smean_215 = memref.load %in_v[%smidx_215] : memref<1511xi32>
        %sscale_215 = memref.load %in_v[%ssidx_215] : memref<1511xi32>
        %dilation_215 = arith.constant 2 : i32
        %padding_215 = arith.constant 8 : i32
        %ki0_215 = arith.constant 1 : i32
        %ki1_215 = arith.constant 6 : i32
        %ki2_215 = arith.constant 8 : i32
        %pad1_215 = arith.constant 0 : i32
        %outidx_215 = arith.constant 215 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_215, %smean_215, %sscale_215, %dilation_215, %padding_215, %ki0_215, %ki1_215, %ki2_215, %pad1_215, %out_v, %outidx_215) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 216: dilation=2 padding=8 kernel=(1,7,8) padding1=1
        %fidx_216 = arith.constant 216 : index
        %bidx_216 = arith.constant 467 : index
        %smidx_216 = arith.constant 887 : index
        %ssidx_216 = arith.constant 1307 : index
        %bias_216 = memref.load %in_v[%bidx_216] : memref<1511xi32>
        %smean_216 = memref.load %in_v[%smidx_216] : memref<1511xi32>
        %sscale_216 = memref.load %in_v[%ssidx_216] : memref<1511xi32>
        %dilation_216 = arith.constant 2 : i32
        %padding_216 = arith.constant 8 : i32
        %ki0_216 = arith.constant 1 : i32
        %ki1_216 = arith.constant 7 : i32
        %ki2_216 = arith.constant 8 : i32
        %pad1_216 = arith.constant 1 : i32
        %outidx_216 = arith.constant 216 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_216, %smean_216, %sscale_216, %dilation_216, %padding_216, %ki0_216, %ki1_216, %ki2_216, %pad1_216, %out_v, %outidx_216) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 217: dilation=2 padding=8 kernel=(2,3,4) padding1=0
        %fidx_217 = arith.constant 217 : index
        %bidx_217 = arith.constant 468 : index
        %smidx_217 = arith.constant 888 : index
        %ssidx_217 = arith.constant 1308 : index
        %bias_217 = memref.load %in_v[%bidx_217] : memref<1511xi32>
        %smean_217 = memref.load %in_v[%smidx_217] : memref<1511xi32>
        %sscale_217 = memref.load %in_v[%ssidx_217] : memref<1511xi32>
        %dilation_217 = arith.constant 2 : i32
        %padding_217 = arith.constant 8 : i32
        %ki0_217 = arith.constant 2 : i32
        %ki1_217 = arith.constant 3 : i32
        %ki2_217 = arith.constant 4 : i32
        %pad1_217 = arith.constant 0 : i32
        %outidx_217 = arith.constant 217 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_217, %smean_217, %sscale_217, %dilation_217, %padding_217, %ki0_217, %ki1_217, %ki2_217, %pad1_217, %out_v, %outidx_217) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 218: dilation=2 padding=8 kernel=(2,3,5) padding1=1
        %fidx_218 = arith.constant 218 : index
        %bidx_218 = arith.constant 469 : index
        %smidx_218 = arith.constant 889 : index
        %ssidx_218 = arith.constant 1309 : index
        %bias_218 = memref.load %in_v[%bidx_218] : memref<1511xi32>
        %smean_218 = memref.load %in_v[%smidx_218] : memref<1511xi32>
        %sscale_218 = memref.load %in_v[%ssidx_218] : memref<1511xi32>
        %dilation_218 = arith.constant 2 : i32
        %padding_218 = arith.constant 8 : i32
        %ki0_218 = arith.constant 2 : i32
        %ki1_218 = arith.constant 3 : i32
        %ki2_218 = arith.constant 5 : i32
        %pad1_218 = arith.constant 1 : i32
        %outidx_218 = arith.constant 218 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_218, %smean_218, %sscale_218, %dilation_218, %padding_218, %ki0_218, %ki1_218, %ki2_218, %pad1_218, %out_v, %outidx_218) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 219: dilation=2 padding=8 kernel=(2,3,6) padding1=0
        %fidx_219 = arith.constant 219 : index
        %bidx_219 = arith.constant 470 : index
        %smidx_219 = arith.constant 890 : index
        %ssidx_219 = arith.constant 1310 : index
        %bias_219 = memref.load %in_v[%bidx_219] : memref<1511xi32>
        %smean_219 = memref.load %in_v[%smidx_219] : memref<1511xi32>
        %sscale_219 = memref.load %in_v[%ssidx_219] : memref<1511xi32>
        %dilation_219 = arith.constant 2 : i32
        %padding_219 = arith.constant 8 : i32
        %ki0_219 = arith.constant 2 : i32
        %ki1_219 = arith.constant 3 : i32
        %ki2_219 = arith.constant 6 : i32
        %pad1_219 = arith.constant 0 : i32
        %outidx_219 = arith.constant 219 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_219, %smean_219, %sscale_219, %dilation_219, %padding_219, %ki0_219, %ki1_219, %ki2_219, %pad1_219, %out_v, %outidx_219) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 220: dilation=2 padding=8 kernel=(2,3,7) padding1=1
        %fidx_220 = arith.constant 220 : index
        %bidx_220 = arith.constant 471 : index
        %smidx_220 = arith.constant 891 : index
        %ssidx_220 = arith.constant 1311 : index
        %bias_220 = memref.load %in_v[%bidx_220] : memref<1511xi32>
        %smean_220 = memref.load %in_v[%smidx_220] : memref<1511xi32>
        %sscale_220 = memref.load %in_v[%ssidx_220] : memref<1511xi32>
        %dilation_220 = arith.constant 2 : i32
        %padding_220 = arith.constant 8 : i32
        %ki0_220 = arith.constant 2 : i32
        %ki1_220 = arith.constant 3 : i32
        %ki2_220 = arith.constant 7 : i32
        %pad1_220 = arith.constant 1 : i32
        %outidx_220 = arith.constant 220 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_220, %smean_220, %sscale_220, %dilation_220, %padding_220, %ki0_220, %ki1_220, %ki2_220, %pad1_220, %out_v, %outidx_220) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 221: dilation=2 padding=8 kernel=(2,3,8) padding1=0
        %fidx_221 = arith.constant 221 : index
        %bidx_221 = arith.constant 472 : index
        %smidx_221 = arith.constant 892 : index
        %ssidx_221 = arith.constant 1312 : index
        %bias_221 = memref.load %in_v[%bidx_221] : memref<1511xi32>
        %smean_221 = memref.load %in_v[%smidx_221] : memref<1511xi32>
        %sscale_221 = memref.load %in_v[%ssidx_221] : memref<1511xi32>
        %dilation_221 = arith.constant 2 : i32
        %padding_221 = arith.constant 8 : i32
        %ki0_221 = arith.constant 2 : i32
        %ki1_221 = arith.constant 3 : i32
        %ki2_221 = arith.constant 8 : i32
        %pad1_221 = arith.constant 0 : i32
        %outidx_221 = arith.constant 221 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_221, %smean_221, %sscale_221, %dilation_221, %padding_221, %ki0_221, %ki1_221, %ki2_221, %pad1_221, %out_v, %outidx_221) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 222: dilation=2 padding=8 kernel=(2,4,5) padding1=1
        %fidx_222 = arith.constant 222 : index
        %bidx_222 = arith.constant 473 : index
        %smidx_222 = arith.constant 893 : index
        %ssidx_222 = arith.constant 1313 : index
        %bias_222 = memref.load %in_v[%bidx_222] : memref<1511xi32>
        %smean_222 = memref.load %in_v[%smidx_222] : memref<1511xi32>
        %sscale_222 = memref.load %in_v[%ssidx_222] : memref<1511xi32>
        %dilation_222 = arith.constant 2 : i32
        %padding_222 = arith.constant 8 : i32
        %ki0_222 = arith.constant 2 : i32
        %ki1_222 = arith.constant 4 : i32
        %ki2_222 = arith.constant 5 : i32
        %pad1_222 = arith.constant 1 : i32
        %outidx_222 = arith.constant 222 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_222, %smean_222, %sscale_222, %dilation_222, %padding_222, %ki0_222, %ki1_222, %ki2_222, %pad1_222, %out_v, %outidx_222) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 223: dilation=2 padding=8 kernel=(2,4,6) padding1=0
        %fidx_223 = arith.constant 223 : index
        %bidx_223 = arith.constant 474 : index
        %smidx_223 = arith.constant 894 : index
        %ssidx_223 = arith.constant 1314 : index
        %bias_223 = memref.load %in_v[%bidx_223] : memref<1511xi32>
        %smean_223 = memref.load %in_v[%smidx_223] : memref<1511xi32>
        %sscale_223 = memref.load %in_v[%ssidx_223] : memref<1511xi32>
        %dilation_223 = arith.constant 2 : i32
        %padding_223 = arith.constant 8 : i32
        %ki0_223 = arith.constant 2 : i32
        %ki1_223 = arith.constant 4 : i32
        %ki2_223 = arith.constant 6 : i32
        %pad1_223 = arith.constant 0 : i32
        %outidx_223 = arith.constant 223 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_223, %smean_223, %sscale_223, %dilation_223, %padding_223, %ki0_223, %ki1_223, %ki2_223, %pad1_223, %out_v, %outidx_223) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 224: dilation=2 padding=8 kernel=(2,4,7) padding1=1
        %fidx_224 = arith.constant 224 : index
        %bidx_224 = arith.constant 475 : index
        %smidx_224 = arith.constant 895 : index
        %ssidx_224 = arith.constant 1315 : index
        %bias_224 = memref.load %in_v[%bidx_224] : memref<1511xi32>
        %smean_224 = memref.load %in_v[%smidx_224] : memref<1511xi32>
        %sscale_224 = memref.load %in_v[%ssidx_224] : memref<1511xi32>
        %dilation_224 = arith.constant 2 : i32
        %padding_224 = arith.constant 8 : i32
        %ki0_224 = arith.constant 2 : i32
        %ki1_224 = arith.constant 4 : i32
        %ki2_224 = arith.constant 7 : i32
        %pad1_224 = arith.constant 1 : i32
        %outidx_224 = arith.constant 224 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_224, %smean_224, %sscale_224, %dilation_224, %padding_224, %ki0_224, %ki1_224, %ki2_224, %pad1_224, %out_v, %outidx_224) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 225: dilation=2 padding=8 kernel=(2,4,8) padding1=0
        %fidx_225 = arith.constant 225 : index
        %bidx_225 = arith.constant 476 : index
        %smidx_225 = arith.constant 896 : index
        %ssidx_225 = arith.constant 1316 : index
        %bias_225 = memref.load %in_v[%bidx_225] : memref<1511xi32>
        %smean_225 = memref.load %in_v[%smidx_225] : memref<1511xi32>
        %sscale_225 = memref.load %in_v[%ssidx_225] : memref<1511xi32>
        %dilation_225 = arith.constant 2 : i32
        %padding_225 = arith.constant 8 : i32
        %ki0_225 = arith.constant 2 : i32
        %ki1_225 = arith.constant 4 : i32
        %ki2_225 = arith.constant 8 : i32
        %pad1_225 = arith.constant 0 : i32
        %outidx_225 = arith.constant 225 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_225, %smean_225, %sscale_225, %dilation_225, %padding_225, %ki0_225, %ki1_225, %ki2_225, %pad1_225, %out_v, %outidx_225) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 226: dilation=2 padding=8 kernel=(2,5,6) padding1=1
        %fidx_226 = arith.constant 226 : index
        %bidx_226 = arith.constant 477 : index
        %smidx_226 = arith.constant 897 : index
        %ssidx_226 = arith.constant 1317 : index
        %bias_226 = memref.load %in_v[%bidx_226] : memref<1511xi32>
        %smean_226 = memref.load %in_v[%smidx_226] : memref<1511xi32>
        %sscale_226 = memref.load %in_v[%ssidx_226] : memref<1511xi32>
        %dilation_226 = arith.constant 2 : i32
        %padding_226 = arith.constant 8 : i32
        %ki0_226 = arith.constant 2 : i32
        %ki1_226 = arith.constant 5 : i32
        %ki2_226 = arith.constant 6 : i32
        %pad1_226 = arith.constant 1 : i32
        %outidx_226 = arith.constant 226 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_226, %smean_226, %sscale_226, %dilation_226, %padding_226, %ki0_226, %ki1_226, %ki2_226, %pad1_226, %out_v, %outidx_226) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 227: dilation=2 padding=8 kernel=(2,5,7) padding1=0
        %fidx_227 = arith.constant 227 : index
        %bidx_227 = arith.constant 478 : index
        %smidx_227 = arith.constant 898 : index
        %ssidx_227 = arith.constant 1318 : index
        %bias_227 = memref.load %in_v[%bidx_227] : memref<1511xi32>
        %smean_227 = memref.load %in_v[%smidx_227] : memref<1511xi32>
        %sscale_227 = memref.load %in_v[%ssidx_227] : memref<1511xi32>
        %dilation_227 = arith.constant 2 : i32
        %padding_227 = arith.constant 8 : i32
        %ki0_227 = arith.constant 2 : i32
        %ki1_227 = arith.constant 5 : i32
        %ki2_227 = arith.constant 7 : i32
        %pad1_227 = arith.constant 0 : i32
        %outidx_227 = arith.constant 227 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_227, %smean_227, %sscale_227, %dilation_227, %padding_227, %ki0_227, %ki1_227, %ki2_227, %pad1_227, %out_v, %outidx_227) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 228: dilation=2 padding=8 kernel=(2,5,8) padding1=1
        %fidx_228 = arith.constant 228 : index
        %bidx_228 = arith.constant 479 : index
        %smidx_228 = arith.constant 899 : index
        %ssidx_228 = arith.constant 1319 : index
        %bias_228 = memref.load %in_v[%bidx_228] : memref<1511xi32>
        %smean_228 = memref.load %in_v[%smidx_228] : memref<1511xi32>
        %sscale_228 = memref.load %in_v[%ssidx_228] : memref<1511xi32>
        %dilation_228 = arith.constant 2 : i32
        %padding_228 = arith.constant 8 : i32
        %ki0_228 = arith.constant 2 : i32
        %ki1_228 = arith.constant 5 : i32
        %ki2_228 = arith.constant 8 : i32
        %pad1_228 = arith.constant 1 : i32
        %outidx_228 = arith.constant 228 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_228, %smean_228, %sscale_228, %dilation_228, %padding_228, %ki0_228, %ki1_228, %ki2_228, %pad1_228, %out_v, %outidx_228) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 229: dilation=2 padding=8 kernel=(2,6,7) padding1=0
        %fidx_229 = arith.constant 229 : index
        %bidx_229 = arith.constant 480 : index
        %smidx_229 = arith.constant 900 : index
        %ssidx_229 = arith.constant 1320 : index
        %bias_229 = memref.load %in_v[%bidx_229] : memref<1511xi32>
        %smean_229 = memref.load %in_v[%smidx_229] : memref<1511xi32>
        %sscale_229 = memref.load %in_v[%ssidx_229] : memref<1511xi32>
        %dilation_229 = arith.constant 2 : i32
        %padding_229 = arith.constant 8 : i32
        %ki0_229 = arith.constant 2 : i32
        %ki1_229 = arith.constant 6 : i32
        %ki2_229 = arith.constant 7 : i32
        %pad1_229 = arith.constant 0 : i32
        %outidx_229 = arith.constant 229 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_229, %smean_229, %sscale_229, %dilation_229, %padding_229, %ki0_229, %ki1_229, %ki2_229, %pad1_229, %out_v, %outidx_229) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 230: dilation=2 padding=8 kernel=(2,6,8) padding1=1
        %fidx_230 = arith.constant 230 : index
        %bidx_230 = arith.constant 481 : index
        %smidx_230 = arith.constant 901 : index
        %ssidx_230 = arith.constant 1321 : index
        %bias_230 = memref.load %in_v[%bidx_230] : memref<1511xi32>
        %smean_230 = memref.load %in_v[%smidx_230] : memref<1511xi32>
        %sscale_230 = memref.load %in_v[%ssidx_230] : memref<1511xi32>
        %dilation_230 = arith.constant 2 : i32
        %padding_230 = arith.constant 8 : i32
        %ki0_230 = arith.constant 2 : i32
        %ki1_230 = arith.constant 6 : i32
        %ki2_230 = arith.constant 8 : i32
        %pad1_230 = arith.constant 1 : i32
        %outidx_230 = arith.constant 230 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_230, %smean_230, %sscale_230, %dilation_230, %padding_230, %ki0_230, %ki1_230, %ki2_230, %pad1_230, %out_v, %outidx_230) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 231: dilation=2 padding=8 kernel=(2,7,8) padding1=0
        %fidx_231 = arith.constant 231 : index
        %bidx_231 = arith.constant 482 : index
        %smidx_231 = arith.constant 902 : index
        %ssidx_231 = arith.constant 1322 : index
        %bias_231 = memref.load %in_v[%bidx_231] : memref<1511xi32>
        %smean_231 = memref.load %in_v[%smidx_231] : memref<1511xi32>
        %sscale_231 = memref.load %in_v[%ssidx_231] : memref<1511xi32>
        %dilation_231 = arith.constant 2 : i32
        %padding_231 = arith.constant 8 : i32
        %ki0_231 = arith.constant 2 : i32
        %ki1_231 = arith.constant 7 : i32
        %ki2_231 = arith.constant 8 : i32
        %pad1_231 = arith.constant 0 : i32
        %outidx_231 = arith.constant 231 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_231, %smean_231, %sscale_231, %dilation_231, %padding_231, %ki0_231, %ki1_231, %ki2_231, %pad1_231, %out_v, %outidx_231) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 232: dilation=2 padding=8 kernel=(3,4,5) padding1=1
        %fidx_232 = arith.constant 232 : index
        %bidx_232 = arith.constant 483 : index
        %smidx_232 = arith.constant 903 : index
        %ssidx_232 = arith.constant 1323 : index
        %bias_232 = memref.load %in_v[%bidx_232] : memref<1511xi32>
        %smean_232 = memref.load %in_v[%smidx_232] : memref<1511xi32>
        %sscale_232 = memref.load %in_v[%ssidx_232] : memref<1511xi32>
        %dilation_232 = arith.constant 2 : i32
        %padding_232 = arith.constant 8 : i32
        %ki0_232 = arith.constant 3 : i32
        %ki1_232 = arith.constant 4 : i32
        %ki2_232 = arith.constant 5 : i32
        %pad1_232 = arith.constant 1 : i32
        %outidx_232 = arith.constant 232 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_232, %smean_232, %sscale_232, %dilation_232, %padding_232, %ki0_232, %ki1_232, %ki2_232, %pad1_232, %out_v, %outidx_232) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 233: dilation=2 padding=8 kernel=(3,4,6) padding1=0
        %fidx_233 = arith.constant 233 : index
        %bidx_233 = arith.constant 484 : index
        %smidx_233 = arith.constant 904 : index
        %ssidx_233 = arith.constant 1324 : index
        %bias_233 = memref.load %in_v[%bidx_233] : memref<1511xi32>
        %smean_233 = memref.load %in_v[%smidx_233] : memref<1511xi32>
        %sscale_233 = memref.load %in_v[%ssidx_233] : memref<1511xi32>
        %dilation_233 = arith.constant 2 : i32
        %padding_233 = arith.constant 8 : i32
        %ki0_233 = arith.constant 3 : i32
        %ki1_233 = arith.constant 4 : i32
        %ki2_233 = arith.constant 6 : i32
        %pad1_233 = arith.constant 0 : i32
        %outidx_233 = arith.constant 233 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_233, %smean_233, %sscale_233, %dilation_233, %padding_233, %ki0_233, %ki1_233, %ki2_233, %pad1_233, %out_v, %outidx_233) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 234: dilation=2 padding=8 kernel=(3,4,7) padding1=1
        %fidx_234 = arith.constant 234 : index
        %bidx_234 = arith.constant 485 : index
        %smidx_234 = arith.constant 905 : index
        %ssidx_234 = arith.constant 1325 : index
        %bias_234 = memref.load %in_v[%bidx_234] : memref<1511xi32>
        %smean_234 = memref.load %in_v[%smidx_234] : memref<1511xi32>
        %sscale_234 = memref.load %in_v[%ssidx_234] : memref<1511xi32>
        %dilation_234 = arith.constant 2 : i32
        %padding_234 = arith.constant 8 : i32
        %ki0_234 = arith.constant 3 : i32
        %ki1_234 = arith.constant 4 : i32
        %ki2_234 = arith.constant 7 : i32
        %pad1_234 = arith.constant 1 : i32
        %outidx_234 = arith.constant 234 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_234, %smean_234, %sscale_234, %dilation_234, %padding_234, %ki0_234, %ki1_234, %ki2_234, %pad1_234, %out_v, %outidx_234) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 235: dilation=2 padding=8 kernel=(3,4,8) padding1=0
        %fidx_235 = arith.constant 235 : index
        %bidx_235 = arith.constant 486 : index
        %smidx_235 = arith.constant 906 : index
        %ssidx_235 = arith.constant 1326 : index
        %bias_235 = memref.load %in_v[%bidx_235] : memref<1511xi32>
        %smean_235 = memref.load %in_v[%smidx_235] : memref<1511xi32>
        %sscale_235 = memref.load %in_v[%ssidx_235] : memref<1511xi32>
        %dilation_235 = arith.constant 2 : i32
        %padding_235 = arith.constant 8 : i32
        %ki0_235 = arith.constant 3 : i32
        %ki1_235 = arith.constant 4 : i32
        %ki2_235 = arith.constant 8 : i32
        %pad1_235 = arith.constant 0 : i32
        %outidx_235 = arith.constant 235 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_235, %smean_235, %sscale_235, %dilation_235, %padding_235, %ki0_235, %ki1_235, %ki2_235, %pad1_235, %out_v, %outidx_235) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 236: dilation=2 padding=8 kernel=(3,5,6) padding1=1
        %fidx_236 = arith.constant 236 : index
        %bidx_236 = arith.constant 487 : index
        %smidx_236 = arith.constant 907 : index
        %ssidx_236 = arith.constant 1327 : index
        %bias_236 = memref.load %in_v[%bidx_236] : memref<1511xi32>
        %smean_236 = memref.load %in_v[%smidx_236] : memref<1511xi32>
        %sscale_236 = memref.load %in_v[%ssidx_236] : memref<1511xi32>
        %dilation_236 = arith.constant 2 : i32
        %padding_236 = arith.constant 8 : i32
        %ki0_236 = arith.constant 3 : i32
        %ki1_236 = arith.constant 5 : i32
        %ki2_236 = arith.constant 6 : i32
        %pad1_236 = arith.constant 1 : i32
        %outidx_236 = arith.constant 236 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_236, %smean_236, %sscale_236, %dilation_236, %padding_236, %ki0_236, %ki1_236, %ki2_236, %pad1_236, %out_v, %outidx_236) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 237: dilation=2 padding=8 kernel=(3,5,7) padding1=0
        %fidx_237 = arith.constant 237 : index
        %bidx_237 = arith.constant 488 : index
        %smidx_237 = arith.constant 908 : index
        %ssidx_237 = arith.constant 1328 : index
        %bias_237 = memref.load %in_v[%bidx_237] : memref<1511xi32>
        %smean_237 = memref.load %in_v[%smidx_237] : memref<1511xi32>
        %sscale_237 = memref.load %in_v[%ssidx_237] : memref<1511xi32>
        %dilation_237 = arith.constant 2 : i32
        %padding_237 = arith.constant 8 : i32
        %ki0_237 = arith.constant 3 : i32
        %ki1_237 = arith.constant 5 : i32
        %ki2_237 = arith.constant 7 : i32
        %pad1_237 = arith.constant 0 : i32
        %outidx_237 = arith.constant 237 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_237, %smean_237, %sscale_237, %dilation_237, %padding_237, %ki0_237, %ki1_237, %ki2_237, %pad1_237, %out_v, %outidx_237) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 238: dilation=2 padding=8 kernel=(3,5,8) padding1=1
        %fidx_238 = arith.constant 238 : index
        %bidx_238 = arith.constant 489 : index
        %smidx_238 = arith.constant 909 : index
        %ssidx_238 = arith.constant 1329 : index
        %bias_238 = memref.load %in_v[%bidx_238] : memref<1511xi32>
        %smean_238 = memref.load %in_v[%smidx_238] : memref<1511xi32>
        %sscale_238 = memref.load %in_v[%ssidx_238] : memref<1511xi32>
        %dilation_238 = arith.constant 2 : i32
        %padding_238 = arith.constant 8 : i32
        %ki0_238 = arith.constant 3 : i32
        %ki1_238 = arith.constant 5 : i32
        %ki2_238 = arith.constant 8 : i32
        %pad1_238 = arith.constant 1 : i32
        %outidx_238 = arith.constant 238 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_238, %smean_238, %sscale_238, %dilation_238, %padding_238, %ki0_238, %ki1_238, %ki2_238, %pad1_238, %out_v, %outidx_238) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 239: dilation=2 padding=8 kernel=(3,6,7) padding1=0
        %fidx_239 = arith.constant 239 : index
        %bidx_239 = arith.constant 490 : index
        %smidx_239 = arith.constant 910 : index
        %ssidx_239 = arith.constant 1330 : index
        %bias_239 = memref.load %in_v[%bidx_239] : memref<1511xi32>
        %smean_239 = memref.load %in_v[%smidx_239] : memref<1511xi32>
        %sscale_239 = memref.load %in_v[%ssidx_239] : memref<1511xi32>
        %dilation_239 = arith.constant 2 : i32
        %padding_239 = arith.constant 8 : i32
        %ki0_239 = arith.constant 3 : i32
        %ki1_239 = arith.constant 6 : i32
        %ki2_239 = arith.constant 7 : i32
        %pad1_239 = arith.constant 0 : i32
        %outidx_239 = arith.constant 239 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_239, %smean_239, %sscale_239, %dilation_239, %padding_239, %ki0_239, %ki1_239, %ki2_239, %pad1_239, %out_v, %outidx_239) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 240: dilation=2 padding=8 kernel=(3,6,8) padding1=1
        %fidx_240 = arith.constant 240 : index
        %bidx_240 = arith.constant 491 : index
        %smidx_240 = arith.constant 911 : index
        %ssidx_240 = arith.constant 1331 : index
        %bias_240 = memref.load %in_v[%bidx_240] : memref<1511xi32>
        %smean_240 = memref.load %in_v[%smidx_240] : memref<1511xi32>
        %sscale_240 = memref.load %in_v[%ssidx_240] : memref<1511xi32>
        %dilation_240 = arith.constant 2 : i32
        %padding_240 = arith.constant 8 : i32
        %ki0_240 = arith.constant 3 : i32
        %ki1_240 = arith.constant 6 : i32
        %ki2_240 = arith.constant 8 : i32
        %pad1_240 = arith.constant 1 : i32
        %outidx_240 = arith.constant 240 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_240, %smean_240, %sscale_240, %dilation_240, %padding_240, %ki0_240, %ki1_240, %ki2_240, %pad1_240, %out_v, %outidx_240) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 241: dilation=2 padding=8 kernel=(3,7,8) padding1=0
        %fidx_241 = arith.constant 241 : index
        %bidx_241 = arith.constant 492 : index
        %smidx_241 = arith.constant 912 : index
        %ssidx_241 = arith.constant 1332 : index
        %bias_241 = memref.load %in_v[%bidx_241] : memref<1511xi32>
        %smean_241 = memref.load %in_v[%smidx_241] : memref<1511xi32>
        %sscale_241 = memref.load %in_v[%ssidx_241] : memref<1511xi32>
        %dilation_241 = arith.constant 2 : i32
        %padding_241 = arith.constant 8 : i32
        %ki0_241 = arith.constant 3 : i32
        %ki1_241 = arith.constant 7 : i32
        %ki2_241 = arith.constant 8 : i32
        %pad1_241 = arith.constant 0 : i32
        %outidx_241 = arith.constant 241 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_241, %smean_241, %sscale_241, %dilation_241, %padding_241, %ki0_241, %ki1_241, %ki2_241, %pad1_241, %out_v, %outidx_241) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 242: dilation=2 padding=8 kernel=(4,5,6) padding1=1
        %fidx_242 = arith.constant 242 : index
        %bidx_242 = arith.constant 493 : index
        %smidx_242 = arith.constant 913 : index
        %ssidx_242 = arith.constant 1333 : index
        %bias_242 = memref.load %in_v[%bidx_242] : memref<1511xi32>
        %smean_242 = memref.load %in_v[%smidx_242] : memref<1511xi32>
        %sscale_242 = memref.load %in_v[%ssidx_242] : memref<1511xi32>
        %dilation_242 = arith.constant 2 : i32
        %padding_242 = arith.constant 8 : i32
        %ki0_242 = arith.constant 4 : i32
        %ki1_242 = arith.constant 5 : i32
        %ki2_242 = arith.constant 6 : i32
        %pad1_242 = arith.constant 1 : i32
        %outidx_242 = arith.constant 242 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_242, %smean_242, %sscale_242, %dilation_242, %padding_242, %ki0_242, %ki1_242, %ki2_242, %pad1_242, %out_v, %outidx_242) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 243: dilation=2 padding=8 kernel=(4,5,7) padding1=0
        %fidx_243 = arith.constant 243 : index
        %bidx_243 = arith.constant 494 : index
        %smidx_243 = arith.constant 914 : index
        %ssidx_243 = arith.constant 1334 : index
        %bias_243 = memref.load %in_v[%bidx_243] : memref<1511xi32>
        %smean_243 = memref.load %in_v[%smidx_243] : memref<1511xi32>
        %sscale_243 = memref.load %in_v[%ssidx_243] : memref<1511xi32>
        %dilation_243 = arith.constant 2 : i32
        %padding_243 = arith.constant 8 : i32
        %ki0_243 = arith.constant 4 : i32
        %ki1_243 = arith.constant 5 : i32
        %ki2_243 = arith.constant 7 : i32
        %pad1_243 = arith.constant 0 : i32
        %outidx_243 = arith.constant 243 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_243, %smean_243, %sscale_243, %dilation_243, %padding_243, %ki0_243, %ki1_243, %ki2_243, %pad1_243, %out_v, %outidx_243) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 244: dilation=2 padding=8 kernel=(4,5,8) padding1=1
        %fidx_244 = arith.constant 244 : index
        %bidx_244 = arith.constant 495 : index
        %smidx_244 = arith.constant 915 : index
        %ssidx_244 = arith.constant 1335 : index
        %bias_244 = memref.load %in_v[%bidx_244] : memref<1511xi32>
        %smean_244 = memref.load %in_v[%smidx_244] : memref<1511xi32>
        %sscale_244 = memref.load %in_v[%ssidx_244] : memref<1511xi32>
        %dilation_244 = arith.constant 2 : i32
        %padding_244 = arith.constant 8 : i32
        %ki0_244 = arith.constant 4 : i32
        %ki1_244 = arith.constant 5 : i32
        %ki2_244 = arith.constant 8 : i32
        %pad1_244 = arith.constant 1 : i32
        %outidx_244 = arith.constant 244 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_244, %smean_244, %sscale_244, %dilation_244, %padding_244, %ki0_244, %ki1_244, %ki2_244, %pad1_244, %out_v, %outidx_244) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 245: dilation=2 padding=8 kernel=(4,6,7) padding1=0
        %fidx_245 = arith.constant 245 : index
        %bidx_245 = arith.constant 496 : index
        %smidx_245 = arith.constant 916 : index
        %ssidx_245 = arith.constant 1336 : index
        %bias_245 = memref.load %in_v[%bidx_245] : memref<1511xi32>
        %smean_245 = memref.load %in_v[%smidx_245] : memref<1511xi32>
        %sscale_245 = memref.load %in_v[%ssidx_245] : memref<1511xi32>
        %dilation_245 = arith.constant 2 : i32
        %padding_245 = arith.constant 8 : i32
        %ki0_245 = arith.constant 4 : i32
        %ki1_245 = arith.constant 6 : i32
        %ki2_245 = arith.constant 7 : i32
        %pad1_245 = arith.constant 0 : i32
        %outidx_245 = arith.constant 245 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_245, %smean_245, %sscale_245, %dilation_245, %padding_245, %ki0_245, %ki1_245, %ki2_245, %pad1_245, %out_v, %outidx_245) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 246: dilation=2 padding=8 kernel=(4,6,8) padding1=1
        %fidx_246 = arith.constant 246 : index
        %bidx_246 = arith.constant 497 : index
        %smidx_246 = arith.constant 917 : index
        %ssidx_246 = arith.constant 1337 : index
        %bias_246 = memref.load %in_v[%bidx_246] : memref<1511xi32>
        %smean_246 = memref.load %in_v[%smidx_246] : memref<1511xi32>
        %sscale_246 = memref.load %in_v[%ssidx_246] : memref<1511xi32>
        %dilation_246 = arith.constant 2 : i32
        %padding_246 = arith.constant 8 : i32
        %ki0_246 = arith.constant 4 : i32
        %ki1_246 = arith.constant 6 : i32
        %ki2_246 = arith.constant 8 : i32
        %pad1_246 = arith.constant 1 : i32
        %outidx_246 = arith.constant 246 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_246, %smean_246, %sscale_246, %dilation_246, %padding_246, %ki0_246, %ki1_246, %ki2_246, %pad1_246, %out_v, %outidx_246) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 247: dilation=2 padding=8 kernel=(4,7,8) padding1=0
        %fidx_247 = arith.constant 247 : index
        %bidx_247 = arith.constant 498 : index
        %smidx_247 = arith.constant 918 : index
        %ssidx_247 = arith.constant 1338 : index
        %bias_247 = memref.load %in_v[%bidx_247] : memref<1511xi32>
        %smean_247 = memref.load %in_v[%smidx_247] : memref<1511xi32>
        %sscale_247 = memref.load %in_v[%ssidx_247] : memref<1511xi32>
        %dilation_247 = arith.constant 2 : i32
        %padding_247 = arith.constant 8 : i32
        %ki0_247 = arith.constant 4 : i32
        %ki1_247 = arith.constant 7 : i32
        %ki2_247 = arith.constant 8 : i32
        %pad1_247 = arith.constant 0 : i32
        %outidx_247 = arith.constant 247 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_247, %smean_247, %sscale_247, %dilation_247, %padding_247, %ki0_247, %ki1_247, %ki2_247, %pad1_247, %out_v, %outidx_247) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 248: dilation=2 padding=8 kernel=(5,6,7) padding1=1
        %fidx_248 = arith.constant 248 : index
        %bidx_248 = arith.constant 499 : index
        %smidx_248 = arith.constant 919 : index
        %ssidx_248 = arith.constant 1339 : index
        %bias_248 = memref.load %in_v[%bidx_248] : memref<1511xi32>
        %smean_248 = memref.load %in_v[%smidx_248] : memref<1511xi32>
        %sscale_248 = memref.load %in_v[%ssidx_248] : memref<1511xi32>
        %dilation_248 = arith.constant 2 : i32
        %padding_248 = arith.constant 8 : i32
        %ki0_248 = arith.constant 5 : i32
        %ki1_248 = arith.constant 6 : i32
        %ki2_248 = arith.constant 7 : i32
        %pad1_248 = arith.constant 1 : i32
        %outidx_248 = arith.constant 248 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_248, %smean_248, %sscale_248, %dilation_248, %padding_248, %ki0_248, %ki1_248, %ki2_248, %pad1_248, %out_v, %outidx_248) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 249: dilation=2 padding=8 kernel=(5,6,8) padding1=0
        %fidx_249 = arith.constant 249 : index
        %bidx_249 = arith.constant 500 : index
        %smidx_249 = arith.constant 920 : index
        %ssidx_249 = arith.constant 1340 : index
        %bias_249 = memref.load %in_v[%bidx_249] : memref<1511xi32>
        %smean_249 = memref.load %in_v[%smidx_249] : memref<1511xi32>
        %sscale_249 = memref.load %in_v[%ssidx_249] : memref<1511xi32>
        %dilation_249 = arith.constant 2 : i32
        %padding_249 = arith.constant 8 : i32
        %ki0_249 = arith.constant 5 : i32
        %ki1_249 = arith.constant 6 : i32
        %ki2_249 = arith.constant 8 : i32
        %pad1_249 = arith.constant 0 : i32
        %outidx_249 = arith.constant 249 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_249, %smean_249, %sscale_249, %dilation_249, %padding_249, %ki0_249, %ki1_249, %ki2_249, %pad1_249, %out_v, %outidx_249) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 250: dilation=2 padding=8 kernel=(5,7,8) padding1=1
        %fidx_250 = arith.constant 250 : index
        %bidx_250 = arith.constant 501 : index
        %smidx_250 = arith.constant 921 : index
        %ssidx_250 = arith.constant 1341 : index
        %bias_250 = memref.load %in_v[%bidx_250] : memref<1511xi32>
        %smean_250 = memref.load %in_v[%smidx_250] : memref<1511xi32>
        %sscale_250 = memref.load %in_v[%ssidx_250] : memref<1511xi32>
        %dilation_250 = arith.constant 2 : i32
        %padding_250 = arith.constant 8 : i32
        %ki0_250 = arith.constant 5 : i32
        %ki1_250 = arith.constant 7 : i32
        %ki2_250 = arith.constant 8 : i32
        %pad1_250 = arith.constant 1 : i32
        %outidx_250 = arith.constant 250 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_250, %smean_250, %sscale_250, %dilation_250, %padding_250, %ki0_250, %ki1_250, %ki2_250, %pad1_250, %out_v, %outidx_250) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 251: dilation=2 padding=8 kernel=(6,7,8) padding1=0
        %fidx_251 = arith.constant 251 : index
        %bidx_251 = arith.constant 502 : index
        %smidx_251 = arith.constant 922 : index
        %ssidx_251 = arith.constant 1342 : index
        %bias_251 = memref.load %in_v[%bidx_251] : memref<1511xi32>
        %smean_251 = memref.load %in_v[%smidx_251] : memref<1511xi32>
        %sscale_251 = memref.load %in_v[%ssidx_251] : memref<1511xi32>
        %dilation_251 = arith.constant 2 : i32
        %padding_251 = arith.constant 8 : i32
        %ki0_251 = arith.constant 6 : i32
        %ki1_251 = arith.constant 7 : i32
        %ki2_251 = arith.constant 8 : i32
        %pad1_251 = arith.constant 0 : i32
        %outidx_251 = arith.constant 251 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_251, %smean_251, %sscale_251, %dilation_251, %padding_251, %ki0_251, %ki1_251, %ki2_251, %pad1_251, %out_v, %outidx_251) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 252: dilation=3 padding=12 kernel=(0,1,2) padding1=0
        %fidx_252 = arith.constant 252 : index
        %bidx_252 = arith.constant 503 : index
        %smidx_252 = arith.constant 923 : index
        %ssidx_252 = arith.constant 1343 : index
        %bias_252 = memref.load %in_v[%bidx_252] : memref<1511xi32>
        %smean_252 = memref.load %in_v[%smidx_252] : memref<1511xi32>
        %sscale_252 = memref.load %in_v[%ssidx_252] : memref<1511xi32>
        %dilation_252 = arith.constant 3 : i32
        %padding_252 = arith.constant 12 : i32
        %ki0_252 = arith.constant 0 : i32
        %ki1_252 = arith.constant 1 : i32
        %ki2_252 = arith.constant 2 : i32
        %pad1_252 = arith.constant 0 : i32
        %outidx_252 = arith.constant 252 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_252, %smean_252, %sscale_252, %dilation_252, %padding_252, %ki0_252, %ki1_252, %ki2_252, %pad1_252, %out_v, %outidx_252) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 253: dilation=3 padding=12 kernel=(0,1,3) padding1=1
        %fidx_253 = arith.constant 253 : index
        %bidx_253 = arith.constant 504 : index
        %smidx_253 = arith.constant 924 : index
        %ssidx_253 = arith.constant 1344 : index
        %bias_253 = memref.load %in_v[%bidx_253] : memref<1511xi32>
        %smean_253 = memref.load %in_v[%smidx_253] : memref<1511xi32>
        %sscale_253 = memref.load %in_v[%ssidx_253] : memref<1511xi32>
        %dilation_253 = arith.constant 3 : i32
        %padding_253 = arith.constant 12 : i32
        %ki0_253 = arith.constant 0 : i32
        %ki1_253 = arith.constant 1 : i32
        %ki2_253 = arith.constant 3 : i32
        %pad1_253 = arith.constant 1 : i32
        %outidx_253 = arith.constant 253 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_253, %smean_253, %sscale_253, %dilation_253, %padding_253, %ki0_253, %ki1_253, %ki2_253, %pad1_253, %out_v, %outidx_253) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 254: dilation=3 padding=12 kernel=(0,1,4) padding1=0
        %fidx_254 = arith.constant 254 : index
        %bidx_254 = arith.constant 505 : index
        %smidx_254 = arith.constant 925 : index
        %ssidx_254 = arith.constant 1345 : index
        %bias_254 = memref.load %in_v[%bidx_254] : memref<1511xi32>
        %smean_254 = memref.load %in_v[%smidx_254] : memref<1511xi32>
        %sscale_254 = memref.load %in_v[%ssidx_254] : memref<1511xi32>
        %dilation_254 = arith.constant 3 : i32
        %padding_254 = arith.constant 12 : i32
        %ki0_254 = arith.constant 0 : i32
        %ki1_254 = arith.constant 1 : i32
        %ki2_254 = arith.constant 4 : i32
        %pad1_254 = arith.constant 0 : i32
        %outidx_254 = arith.constant 254 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_254, %smean_254, %sscale_254, %dilation_254, %padding_254, %ki0_254, %ki1_254, %ki2_254, %pad1_254, %out_v, %outidx_254) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 255: dilation=3 padding=12 kernel=(0,1,5) padding1=1
        %fidx_255 = arith.constant 255 : index
        %bidx_255 = arith.constant 506 : index
        %smidx_255 = arith.constant 926 : index
        %ssidx_255 = arith.constant 1346 : index
        %bias_255 = memref.load %in_v[%bidx_255] : memref<1511xi32>
        %smean_255 = memref.load %in_v[%smidx_255] : memref<1511xi32>
        %sscale_255 = memref.load %in_v[%ssidx_255] : memref<1511xi32>
        %dilation_255 = arith.constant 3 : i32
        %padding_255 = arith.constant 12 : i32
        %ki0_255 = arith.constant 0 : i32
        %ki1_255 = arith.constant 1 : i32
        %ki2_255 = arith.constant 5 : i32
        %pad1_255 = arith.constant 1 : i32
        %outidx_255 = arith.constant 255 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_255, %smean_255, %sscale_255, %dilation_255, %padding_255, %ki0_255, %ki1_255, %ki2_255, %pad1_255, %out_v, %outidx_255) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 256: dilation=3 padding=12 kernel=(0,1,6) padding1=0
        %fidx_256 = arith.constant 256 : index
        %bidx_256 = arith.constant 507 : index
        %smidx_256 = arith.constant 927 : index
        %ssidx_256 = arith.constant 1347 : index
        %bias_256 = memref.load %in_v[%bidx_256] : memref<1511xi32>
        %smean_256 = memref.load %in_v[%smidx_256] : memref<1511xi32>
        %sscale_256 = memref.load %in_v[%ssidx_256] : memref<1511xi32>
        %dilation_256 = arith.constant 3 : i32
        %padding_256 = arith.constant 12 : i32
        %ki0_256 = arith.constant 0 : i32
        %ki1_256 = arith.constant 1 : i32
        %ki2_256 = arith.constant 6 : i32
        %pad1_256 = arith.constant 0 : i32
        %outidx_256 = arith.constant 256 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_256, %smean_256, %sscale_256, %dilation_256, %padding_256, %ki0_256, %ki1_256, %ki2_256, %pad1_256, %out_v, %outidx_256) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 257: dilation=3 padding=12 kernel=(0,1,7) padding1=1
        %fidx_257 = arith.constant 257 : index
        %bidx_257 = arith.constant 508 : index
        %smidx_257 = arith.constant 928 : index
        %ssidx_257 = arith.constant 1348 : index
        %bias_257 = memref.load %in_v[%bidx_257] : memref<1511xi32>
        %smean_257 = memref.load %in_v[%smidx_257] : memref<1511xi32>
        %sscale_257 = memref.load %in_v[%ssidx_257] : memref<1511xi32>
        %dilation_257 = arith.constant 3 : i32
        %padding_257 = arith.constant 12 : i32
        %ki0_257 = arith.constant 0 : i32
        %ki1_257 = arith.constant 1 : i32
        %ki2_257 = arith.constant 7 : i32
        %pad1_257 = arith.constant 1 : i32
        %outidx_257 = arith.constant 257 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_257, %smean_257, %sscale_257, %dilation_257, %padding_257, %ki0_257, %ki1_257, %ki2_257, %pad1_257, %out_v, %outidx_257) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 258: dilation=3 padding=12 kernel=(0,1,8) padding1=0
        %fidx_258 = arith.constant 258 : index
        %bidx_258 = arith.constant 509 : index
        %smidx_258 = arith.constant 929 : index
        %ssidx_258 = arith.constant 1349 : index
        %bias_258 = memref.load %in_v[%bidx_258] : memref<1511xi32>
        %smean_258 = memref.load %in_v[%smidx_258] : memref<1511xi32>
        %sscale_258 = memref.load %in_v[%ssidx_258] : memref<1511xi32>
        %dilation_258 = arith.constant 3 : i32
        %padding_258 = arith.constant 12 : i32
        %ki0_258 = arith.constant 0 : i32
        %ki1_258 = arith.constant 1 : i32
        %ki2_258 = arith.constant 8 : i32
        %pad1_258 = arith.constant 0 : i32
        %outidx_258 = arith.constant 258 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_258, %smean_258, %sscale_258, %dilation_258, %padding_258, %ki0_258, %ki1_258, %ki2_258, %pad1_258, %out_v, %outidx_258) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 259: dilation=3 padding=12 kernel=(0,2,3) padding1=1
        %fidx_259 = arith.constant 259 : index
        %bidx_259 = arith.constant 510 : index
        %smidx_259 = arith.constant 930 : index
        %ssidx_259 = arith.constant 1350 : index
        %bias_259 = memref.load %in_v[%bidx_259] : memref<1511xi32>
        %smean_259 = memref.load %in_v[%smidx_259] : memref<1511xi32>
        %sscale_259 = memref.load %in_v[%ssidx_259] : memref<1511xi32>
        %dilation_259 = arith.constant 3 : i32
        %padding_259 = arith.constant 12 : i32
        %ki0_259 = arith.constant 0 : i32
        %ki1_259 = arith.constant 2 : i32
        %ki2_259 = arith.constant 3 : i32
        %pad1_259 = arith.constant 1 : i32
        %outidx_259 = arith.constant 259 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_259, %smean_259, %sscale_259, %dilation_259, %padding_259, %ki0_259, %ki1_259, %ki2_259, %pad1_259, %out_v, %outidx_259) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 260: dilation=3 padding=12 kernel=(0,2,4) padding1=0
        %fidx_260 = arith.constant 260 : index
        %bidx_260 = arith.constant 511 : index
        %smidx_260 = arith.constant 931 : index
        %ssidx_260 = arith.constant 1351 : index
        %bias_260 = memref.load %in_v[%bidx_260] : memref<1511xi32>
        %smean_260 = memref.load %in_v[%smidx_260] : memref<1511xi32>
        %sscale_260 = memref.load %in_v[%ssidx_260] : memref<1511xi32>
        %dilation_260 = arith.constant 3 : i32
        %padding_260 = arith.constant 12 : i32
        %ki0_260 = arith.constant 0 : i32
        %ki1_260 = arith.constant 2 : i32
        %ki2_260 = arith.constant 4 : i32
        %pad1_260 = arith.constant 0 : i32
        %outidx_260 = arith.constant 260 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_260, %smean_260, %sscale_260, %dilation_260, %padding_260, %ki0_260, %ki1_260, %ki2_260, %pad1_260, %out_v, %outidx_260) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 261: dilation=3 padding=12 kernel=(0,2,5) padding1=1
        %fidx_261 = arith.constant 261 : index
        %bidx_261 = arith.constant 512 : index
        %smidx_261 = arith.constant 932 : index
        %ssidx_261 = arith.constant 1352 : index
        %bias_261 = memref.load %in_v[%bidx_261] : memref<1511xi32>
        %smean_261 = memref.load %in_v[%smidx_261] : memref<1511xi32>
        %sscale_261 = memref.load %in_v[%ssidx_261] : memref<1511xi32>
        %dilation_261 = arith.constant 3 : i32
        %padding_261 = arith.constant 12 : i32
        %ki0_261 = arith.constant 0 : i32
        %ki1_261 = arith.constant 2 : i32
        %ki2_261 = arith.constant 5 : i32
        %pad1_261 = arith.constant 1 : i32
        %outidx_261 = arith.constant 261 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_261, %smean_261, %sscale_261, %dilation_261, %padding_261, %ki0_261, %ki1_261, %ki2_261, %pad1_261, %out_v, %outidx_261) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 262: dilation=3 padding=12 kernel=(0,2,6) padding1=0
        %fidx_262 = arith.constant 262 : index
        %bidx_262 = arith.constant 513 : index
        %smidx_262 = arith.constant 933 : index
        %ssidx_262 = arith.constant 1353 : index
        %bias_262 = memref.load %in_v[%bidx_262] : memref<1511xi32>
        %smean_262 = memref.load %in_v[%smidx_262] : memref<1511xi32>
        %sscale_262 = memref.load %in_v[%ssidx_262] : memref<1511xi32>
        %dilation_262 = arith.constant 3 : i32
        %padding_262 = arith.constant 12 : i32
        %ki0_262 = arith.constant 0 : i32
        %ki1_262 = arith.constant 2 : i32
        %ki2_262 = arith.constant 6 : i32
        %pad1_262 = arith.constant 0 : i32
        %outidx_262 = arith.constant 262 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_262, %smean_262, %sscale_262, %dilation_262, %padding_262, %ki0_262, %ki1_262, %ki2_262, %pad1_262, %out_v, %outidx_262) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 263: dilation=3 padding=12 kernel=(0,2,7) padding1=1
        %fidx_263 = arith.constant 263 : index
        %bidx_263 = arith.constant 514 : index
        %smidx_263 = arith.constant 934 : index
        %ssidx_263 = arith.constant 1354 : index
        %bias_263 = memref.load %in_v[%bidx_263] : memref<1511xi32>
        %smean_263 = memref.load %in_v[%smidx_263] : memref<1511xi32>
        %sscale_263 = memref.load %in_v[%ssidx_263] : memref<1511xi32>
        %dilation_263 = arith.constant 3 : i32
        %padding_263 = arith.constant 12 : i32
        %ki0_263 = arith.constant 0 : i32
        %ki1_263 = arith.constant 2 : i32
        %ki2_263 = arith.constant 7 : i32
        %pad1_263 = arith.constant 1 : i32
        %outidx_263 = arith.constant 263 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_263, %smean_263, %sscale_263, %dilation_263, %padding_263, %ki0_263, %ki1_263, %ki2_263, %pad1_263, %out_v, %outidx_263) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 264: dilation=3 padding=12 kernel=(0,2,8) padding1=0
        %fidx_264 = arith.constant 264 : index
        %bidx_264 = arith.constant 515 : index
        %smidx_264 = arith.constant 935 : index
        %ssidx_264 = arith.constant 1355 : index
        %bias_264 = memref.load %in_v[%bidx_264] : memref<1511xi32>
        %smean_264 = memref.load %in_v[%smidx_264] : memref<1511xi32>
        %sscale_264 = memref.load %in_v[%ssidx_264] : memref<1511xi32>
        %dilation_264 = arith.constant 3 : i32
        %padding_264 = arith.constant 12 : i32
        %ki0_264 = arith.constant 0 : i32
        %ki1_264 = arith.constant 2 : i32
        %ki2_264 = arith.constant 8 : i32
        %pad1_264 = arith.constant 0 : i32
        %outidx_264 = arith.constant 264 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_264, %smean_264, %sscale_264, %dilation_264, %padding_264, %ki0_264, %ki1_264, %ki2_264, %pad1_264, %out_v, %outidx_264) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 265: dilation=3 padding=12 kernel=(0,3,4) padding1=1
        %fidx_265 = arith.constant 265 : index
        %bidx_265 = arith.constant 516 : index
        %smidx_265 = arith.constant 936 : index
        %ssidx_265 = arith.constant 1356 : index
        %bias_265 = memref.load %in_v[%bidx_265] : memref<1511xi32>
        %smean_265 = memref.load %in_v[%smidx_265] : memref<1511xi32>
        %sscale_265 = memref.load %in_v[%ssidx_265] : memref<1511xi32>
        %dilation_265 = arith.constant 3 : i32
        %padding_265 = arith.constant 12 : i32
        %ki0_265 = arith.constant 0 : i32
        %ki1_265 = arith.constant 3 : i32
        %ki2_265 = arith.constant 4 : i32
        %pad1_265 = arith.constant 1 : i32
        %outidx_265 = arith.constant 265 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_265, %smean_265, %sscale_265, %dilation_265, %padding_265, %ki0_265, %ki1_265, %ki2_265, %pad1_265, %out_v, %outidx_265) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 266: dilation=3 padding=12 kernel=(0,3,5) padding1=0
        %fidx_266 = arith.constant 266 : index
        %bidx_266 = arith.constant 517 : index
        %smidx_266 = arith.constant 937 : index
        %ssidx_266 = arith.constant 1357 : index
        %bias_266 = memref.load %in_v[%bidx_266] : memref<1511xi32>
        %smean_266 = memref.load %in_v[%smidx_266] : memref<1511xi32>
        %sscale_266 = memref.load %in_v[%ssidx_266] : memref<1511xi32>
        %dilation_266 = arith.constant 3 : i32
        %padding_266 = arith.constant 12 : i32
        %ki0_266 = arith.constant 0 : i32
        %ki1_266 = arith.constant 3 : i32
        %ki2_266 = arith.constant 5 : i32
        %pad1_266 = arith.constant 0 : i32
        %outidx_266 = arith.constant 266 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_266, %smean_266, %sscale_266, %dilation_266, %padding_266, %ki0_266, %ki1_266, %ki2_266, %pad1_266, %out_v, %outidx_266) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 267: dilation=3 padding=12 kernel=(0,3,6) padding1=1
        %fidx_267 = arith.constant 267 : index
        %bidx_267 = arith.constant 518 : index
        %smidx_267 = arith.constant 938 : index
        %ssidx_267 = arith.constant 1358 : index
        %bias_267 = memref.load %in_v[%bidx_267] : memref<1511xi32>
        %smean_267 = memref.load %in_v[%smidx_267] : memref<1511xi32>
        %sscale_267 = memref.load %in_v[%ssidx_267] : memref<1511xi32>
        %dilation_267 = arith.constant 3 : i32
        %padding_267 = arith.constant 12 : i32
        %ki0_267 = arith.constant 0 : i32
        %ki1_267 = arith.constant 3 : i32
        %ki2_267 = arith.constant 6 : i32
        %pad1_267 = arith.constant 1 : i32
        %outidx_267 = arith.constant 267 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_267, %smean_267, %sscale_267, %dilation_267, %padding_267, %ki0_267, %ki1_267, %ki2_267, %pad1_267, %out_v, %outidx_267) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 268: dilation=3 padding=12 kernel=(0,3,7) padding1=0
        %fidx_268 = arith.constant 268 : index
        %bidx_268 = arith.constant 519 : index
        %smidx_268 = arith.constant 939 : index
        %ssidx_268 = arith.constant 1359 : index
        %bias_268 = memref.load %in_v[%bidx_268] : memref<1511xi32>
        %smean_268 = memref.load %in_v[%smidx_268] : memref<1511xi32>
        %sscale_268 = memref.load %in_v[%ssidx_268] : memref<1511xi32>
        %dilation_268 = arith.constant 3 : i32
        %padding_268 = arith.constant 12 : i32
        %ki0_268 = arith.constant 0 : i32
        %ki1_268 = arith.constant 3 : i32
        %ki2_268 = arith.constant 7 : i32
        %pad1_268 = arith.constant 0 : i32
        %outidx_268 = arith.constant 268 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_268, %smean_268, %sscale_268, %dilation_268, %padding_268, %ki0_268, %ki1_268, %ki2_268, %pad1_268, %out_v, %outidx_268) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 269: dilation=3 padding=12 kernel=(0,3,8) padding1=1
        %fidx_269 = arith.constant 269 : index
        %bidx_269 = arith.constant 520 : index
        %smidx_269 = arith.constant 940 : index
        %ssidx_269 = arith.constant 1360 : index
        %bias_269 = memref.load %in_v[%bidx_269] : memref<1511xi32>
        %smean_269 = memref.load %in_v[%smidx_269] : memref<1511xi32>
        %sscale_269 = memref.load %in_v[%ssidx_269] : memref<1511xi32>
        %dilation_269 = arith.constant 3 : i32
        %padding_269 = arith.constant 12 : i32
        %ki0_269 = arith.constant 0 : i32
        %ki1_269 = arith.constant 3 : i32
        %ki2_269 = arith.constant 8 : i32
        %pad1_269 = arith.constant 1 : i32
        %outidx_269 = arith.constant 269 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_269, %smean_269, %sscale_269, %dilation_269, %padding_269, %ki0_269, %ki1_269, %ki2_269, %pad1_269, %out_v, %outidx_269) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 270: dilation=3 padding=12 kernel=(0,4,5) padding1=0
        %fidx_270 = arith.constant 270 : index
        %bidx_270 = arith.constant 521 : index
        %smidx_270 = arith.constant 941 : index
        %ssidx_270 = arith.constant 1361 : index
        %bias_270 = memref.load %in_v[%bidx_270] : memref<1511xi32>
        %smean_270 = memref.load %in_v[%smidx_270] : memref<1511xi32>
        %sscale_270 = memref.load %in_v[%ssidx_270] : memref<1511xi32>
        %dilation_270 = arith.constant 3 : i32
        %padding_270 = arith.constant 12 : i32
        %ki0_270 = arith.constant 0 : i32
        %ki1_270 = arith.constant 4 : i32
        %ki2_270 = arith.constant 5 : i32
        %pad1_270 = arith.constant 0 : i32
        %outidx_270 = arith.constant 270 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_270, %smean_270, %sscale_270, %dilation_270, %padding_270, %ki0_270, %ki1_270, %ki2_270, %pad1_270, %out_v, %outidx_270) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 271: dilation=3 padding=12 kernel=(0,4,6) padding1=1
        %fidx_271 = arith.constant 271 : index
        %bidx_271 = arith.constant 522 : index
        %smidx_271 = arith.constant 942 : index
        %ssidx_271 = arith.constant 1362 : index
        %bias_271 = memref.load %in_v[%bidx_271] : memref<1511xi32>
        %smean_271 = memref.load %in_v[%smidx_271] : memref<1511xi32>
        %sscale_271 = memref.load %in_v[%ssidx_271] : memref<1511xi32>
        %dilation_271 = arith.constant 3 : i32
        %padding_271 = arith.constant 12 : i32
        %ki0_271 = arith.constant 0 : i32
        %ki1_271 = arith.constant 4 : i32
        %ki2_271 = arith.constant 6 : i32
        %pad1_271 = arith.constant 1 : i32
        %outidx_271 = arith.constant 271 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_271, %smean_271, %sscale_271, %dilation_271, %padding_271, %ki0_271, %ki1_271, %ki2_271, %pad1_271, %out_v, %outidx_271) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 272: dilation=3 padding=12 kernel=(0,4,7) padding1=0
        %fidx_272 = arith.constant 272 : index
        %bidx_272 = arith.constant 523 : index
        %smidx_272 = arith.constant 943 : index
        %ssidx_272 = arith.constant 1363 : index
        %bias_272 = memref.load %in_v[%bidx_272] : memref<1511xi32>
        %smean_272 = memref.load %in_v[%smidx_272] : memref<1511xi32>
        %sscale_272 = memref.load %in_v[%ssidx_272] : memref<1511xi32>
        %dilation_272 = arith.constant 3 : i32
        %padding_272 = arith.constant 12 : i32
        %ki0_272 = arith.constant 0 : i32
        %ki1_272 = arith.constant 4 : i32
        %ki2_272 = arith.constant 7 : i32
        %pad1_272 = arith.constant 0 : i32
        %outidx_272 = arith.constant 272 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_272, %smean_272, %sscale_272, %dilation_272, %padding_272, %ki0_272, %ki1_272, %ki2_272, %pad1_272, %out_v, %outidx_272) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 273: dilation=3 padding=12 kernel=(0,4,8) padding1=1
        %fidx_273 = arith.constant 273 : index
        %bidx_273 = arith.constant 524 : index
        %smidx_273 = arith.constant 944 : index
        %ssidx_273 = arith.constant 1364 : index
        %bias_273 = memref.load %in_v[%bidx_273] : memref<1511xi32>
        %smean_273 = memref.load %in_v[%smidx_273] : memref<1511xi32>
        %sscale_273 = memref.load %in_v[%ssidx_273] : memref<1511xi32>
        %dilation_273 = arith.constant 3 : i32
        %padding_273 = arith.constant 12 : i32
        %ki0_273 = arith.constant 0 : i32
        %ki1_273 = arith.constant 4 : i32
        %ki2_273 = arith.constant 8 : i32
        %pad1_273 = arith.constant 1 : i32
        %outidx_273 = arith.constant 273 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_273, %smean_273, %sscale_273, %dilation_273, %padding_273, %ki0_273, %ki1_273, %ki2_273, %pad1_273, %out_v, %outidx_273) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 274: dilation=3 padding=12 kernel=(0,5,6) padding1=0
        %fidx_274 = arith.constant 274 : index
        %bidx_274 = arith.constant 525 : index
        %smidx_274 = arith.constant 945 : index
        %ssidx_274 = arith.constant 1365 : index
        %bias_274 = memref.load %in_v[%bidx_274] : memref<1511xi32>
        %smean_274 = memref.load %in_v[%smidx_274] : memref<1511xi32>
        %sscale_274 = memref.load %in_v[%ssidx_274] : memref<1511xi32>
        %dilation_274 = arith.constant 3 : i32
        %padding_274 = arith.constant 12 : i32
        %ki0_274 = arith.constant 0 : i32
        %ki1_274 = arith.constant 5 : i32
        %ki2_274 = arith.constant 6 : i32
        %pad1_274 = arith.constant 0 : i32
        %outidx_274 = arith.constant 274 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_274, %smean_274, %sscale_274, %dilation_274, %padding_274, %ki0_274, %ki1_274, %ki2_274, %pad1_274, %out_v, %outidx_274) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 275: dilation=3 padding=12 kernel=(0,5,7) padding1=1
        %fidx_275 = arith.constant 275 : index
        %bidx_275 = arith.constant 526 : index
        %smidx_275 = arith.constant 946 : index
        %ssidx_275 = arith.constant 1366 : index
        %bias_275 = memref.load %in_v[%bidx_275] : memref<1511xi32>
        %smean_275 = memref.load %in_v[%smidx_275] : memref<1511xi32>
        %sscale_275 = memref.load %in_v[%ssidx_275] : memref<1511xi32>
        %dilation_275 = arith.constant 3 : i32
        %padding_275 = arith.constant 12 : i32
        %ki0_275 = arith.constant 0 : i32
        %ki1_275 = arith.constant 5 : i32
        %ki2_275 = arith.constant 7 : i32
        %pad1_275 = arith.constant 1 : i32
        %outidx_275 = arith.constant 275 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_275, %smean_275, %sscale_275, %dilation_275, %padding_275, %ki0_275, %ki1_275, %ki2_275, %pad1_275, %out_v, %outidx_275) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 276: dilation=3 padding=12 kernel=(0,5,8) padding1=0
        %fidx_276 = arith.constant 276 : index
        %bidx_276 = arith.constant 527 : index
        %smidx_276 = arith.constant 947 : index
        %ssidx_276 = arith.constant 1367 : index
        %bias_276 = memref.load %in_v[%bidx_276] : memref<1511xi32>
        %smean_276 = memref.load %in_v[%smidx_276] : memref<1511xi32>
        %sscale_276 = memref.load %in_v[%ssidx_276] : memref<1511xi32>
        %dilation_276 = arith.constant 3 : i32
        %padding_276 = arith.constant 12 : i32
        %ki0_276 = arith.constant 0 : i32
        %ki1_276 = arith.constant 5 : i32
        %ki2_276 = arith.constant 8 : i32
        %pad1_276 = arith.constant 0 : i32
        %outidx_276 = arith.constant 276 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_276, %smean_276, %sscale_276, %dilation_276, %padding_276, %ki0_276, %ki1_276, %ki2_276, %pad1_276, %out_v, %outidx_276) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 277: dilation=3 padding=12 kernel=(0,6,7) padding1=1
        %fidx_277 = arith.constant 277 : index
        %bidx_277 = arith.constant 528 : index
        %smidx_277 = arith.constant 948 : index
        %ssidx_277 = arith.constant 1368 : index
        %bias_277 = memref.load %in_v[%bidx_277] : memref<1511xi32>
        %smean_277 = memref.load %in_v[%smidx_277] : memref<1511xi32>
        %sscale_277 = memref.load %in_v[%ssidx_277] : memref<1511xi32>
        %dilation_277 = arith.constant 3 : i32
        %padding_277 = arith.constant 12 : i32
        %ki0_277 = arith.constant 0 : i32
        %ki1_277 = arith.constant 6 : i32
        %ki2_277 = arith.constant 7 : i32
        %pad1_277 = arith.constant 1 : i32
        %outidx_277 = arith.constant 277 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_277, %smean_277, %sscale_277, %dilation_277, %padding_277, %ki0_277, %ki1_277, %ki2_277, %pad1_277, %out_v, %outidx_277) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 278: dilation=3 padding=12 kernel=(0,6,8) padding1=0
        %fidx_278 = arith.constant 278 : index
        %bidx_278 = arith.constant 529 : index
        %smidx_278 = arith.constant 949 : index
        %ssidx_278 = arith.constant 1369 : index
        %bias_278 = memref.load %in_v[%bidx_278] : memref<1511xi32>
        %smean_278 = memref.load %in_v[%smidx_278] : memref<1511xi32>
        %sscale_278 = memref.load %in_v[%ssidx_278] : memref<1511xi32>
        %dilation_278 = arith.constant 3 : i32
        %padding_278 = arith.constant 12 : i32
        %ki0_278 = arith.constant 0 : i32
        %ki1_278 = arith.constant 6 : i32
        %ki2_278 = arith.constant 8 : i32
        %pad1_278 = arith.constant 0 : i32
        %outidx_278 = arith.constant 278 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_278, %smean_278, %sscale_278, %dilation_278, %padding_278, %ki0_278, %ki1_278, %ki2_278, %pad1_278, %out_v, %outidx_278) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 279: dilation=3 padding=12 kernel=(0,7,8) padding1=1
        %fidx_279 = arith.constant 279 : index
        %bidx_279 = arith.constant 530 : index
        %smidx_279 = arith.constant 950 : index
        %ssidx_279 = arith.constant 1370 : index
        %bias_279 = memref.load %in_v[%bidx_279] : memref<1511xi32>
        %smean_279 = memref.load %in_v[%smidx_279] : memref<1511xi32>
        %sscale_279 = memref.load %in_v[%ssidx_279] : memref<1511xi32>
        %dilation_279 = arith.constant 3 : i32
        %padding_279 = arith.constant 12 : i32
        %ki0_279 = arith.constant 0 : i32
        %ki1_279 = arith.constant 7 : i32
        %ki2_279 = arith.constant 8 : i32
        %pad1_279 = arith.constant 1 : i32
        %outidx_279 = arith.constant 279 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_279, %smean_279, %sscale_279, %dilation_279, %padding_279, %ki0_279, %ki1_279, %ki2_279, %pad1_279, %out_v, %outidx_279) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 280: dilation=3 padding=12 kernel=(1,2,3) padding1=0
        %fidx_280 = arith.constant 280 : index
        %bidx_280 = arith.constant 531 : index
        %smidx_280 = arith.constant 951 : index
        %ssidx_280 = arith.constant 1371 : index
        %bias_280 = memref.load %in_v[%bidx_280] : memref<1511xi32>
        %smean_280 = memref.load %in_v[%smidx_280] : memref<1511xi32>
        %sscale_280 = memref.load %in_v[%ssidx_280] : memref<1511xi32>
        %dilation_280 = arith.constant 3 : i32
        %padding_280 = arith.constant 12 : i32
        %ki0_280 = arith.constant 1 : i32
        %ki1_280 = arith.constant 2 : i32
        %ki2_280 = arith.constant 3 : i32
        %pad1_280 = arith.constant 0 : i32
        %outidx_280 = arith.constant 280 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_280, %smean_280, %sscale_280, %dilation_280, %padding_280, %ki0_280, %ki1_280, %ki2_280, %pad1_280, %out_v, %outidx_280) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 281: dilation=3 padding=12 kernel=(1,2,4) padding1=1
        %fidx_281 = arith.constant 281 : index
        %bidx_281 = arith.constant 532 : index
        %smidx_281 = arith.constant 952 : index
        %ssidx_281 = arith.constant 1372 : index
        %bias_281 = memref.load %in_v[%bidx_281] : memref<1511xi32>
        %smean_281 = memref.load %in_v[%smidx_281] : memref<1511xi32>
        %sscale_281 = memref.load %in_v[%ssidx_281] : memref<1511xi32>
        %dilation_281 = arith.constant 3 : i32
        %padding_281 = arith.constant 12 : i32
        %ki0_281 = arith.constant 1 : i32
        %ki1_281 = arith.constant 2 : i32
        %ki2_281 = arith.constant 4 : i32
        %pad1_281 = arith.constant 1 : i32
        %outidx_281 = arith.constant 281 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_281, %smean_281, %sscale_281, %dilation_281, %padding_281, %ki0_281, %ki1_281, %ki2_281, %pad1_281, %out_v, %outidx_281) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 282: dilation=3 padding=12 kernel=(1,2,5) padding1=0
        %fidx_282 = arith.constant 282 : index
        %bidx_282 = arith.constant 533 : index
        %smidx_282 = arith.constant 953 : index
        %ssidx_282 = arith.constant 1373 : index
        %bias_282 = memref.load %in_v[%bidx_282] : memref<1511xi32>
        %smean_282 = memref.load %in_v[%smidx_282] : memref<1511xi32>
        %sscale_282 = memref.load %in_v[%ssidx_282] : memref<1511xi32>
        %dilation_282 = arith.constant 3 : i32
        %padding_282 = arith.constant 12 : i32
        %ki0_282 = arith.constant 1 : i32
        %ki1_282 = arith.constant 2 : i32
        %ki2_282 = arith.constant 5 : i32
        %pad1_282 = arith.constant 0 : i32
        %outidx_282 = arith.constant 282 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_282, %smean_282, %sscale_282, %dilation_282, %padding_282, %ki0_282, %ki1_282, %ki2_282, %pad1_282, %out_v, %outidx_282) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 283: dilation=3 padding=12 kernel=(1,2,6) padding1=1
        %fidx_283 = arith.constant 283 : index
        %bidx_283 = arith.constant 534 : index
        %smidx_283 = arith.constant 954 : index
        %ssidx_283 = arith.constant 1374 : index
        %bias_283 = memref.load %in_v[%bidx_283] : memref<1511xi32>
        %smean_283 = memref.load %in_v[%smidx_283] : memref<1511xi32>
        %sscale_283 = memref.load %in_v[%ssidx_283] : memref<1511xi32>
        %dilation_283 = arith.constant 3 : i32
        %padding_283 = arith.constant 12 : i32
        %ki0_283 = arith.constant 1 : i32
        %ki1_283 = arith.constant 2 : i32
        %ki2_283 = arith.constant 6 : i32
        %pad1_283 = arith.constant 1 : i32
        %outidx_283 = arith.constant 283 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_283, %smean_283, %sscale_283, %dilation_283, %padding_283, %ki0_283, %ki1_283, %ki2_283, %pad1_283, %out_v, %outidx_283) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 284: dilation=3 padding=12 kernel=(1,2,7) padding1=0
        %fidx_284 = arith.constant 284 : index
        %bidx_284 = arith.constant 535 : index
        %smidx_284 = arith.constant 955 : index
        %ssidx_284 = arith.constant 1375 : index
        %bias_284 = memref.load %in_v[%bidx_284] : memref<1511xi32>
        %smean_284 = memref.load %in_v[%smidx_284] : memref<1511xi32>
        %sscale_284 = memref.load %in_v[%ssidx_284] : memref<1511xi32>
        %dilation_284 = arith.constant 3 : i32
        %padding_284 = arith.constant 12 : i32
        %ki0_284 = arith.constant 1 : i32
        %ki1_284 = arith.constant 2 : i32
        %ki2_284 = arith.constant 7 : i32
        %pad1_284 = arith.constant 0 : i32
        %outidx_284 = arith.constant 284 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_284, %smean_284, %sscale_284, %dilation_284, %padding_284, %ki0_284, %ki1_284, %ki2_284, %pad1_284, %out_v, %outidx_284) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 285: dilation=3 padding=12 kernel=(1,2,8) padding1=1
        %fidx_285 = arith.constant 285 : index
        %bidx_285 = arith.constant 536 : index
        %smidx_285 = arith.constant 956 : index
        %ssidx_285 = arith.constant 1376 : index
        %bias_285 = memref.load %in_v[%bidx_285] : memref<1511xi32>
        %smean_285 = memref.load %in_v[%smidx_285] : memref<1511xi32>
        %sscale_285 = memref.load %in_v[%ssidx_285] : memref<1511xi32>
        %dilation_285 = arith.constant 3 : i32
        %padding_285 = arith.constant 12 : i32
        %ki0_285 = arith.constant 1 : i32
        %ki1_285 = arith.constant 2 : i32
        %ki2_285 = arith.constant 8 : i32
        %pad1_285 = arith.constant 1 : i32
        %outidx_285 = arith.constant 285 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_285, %smean_285, %sscale_285, %dilation_285, %padding_285, %ki0_285, %ki1_285, %ki2_285, %pad1_285, %out_v, %outidx_285) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 286: dilation=3 padding=12 kernel=(1,3,4) padding1=0
        %fidx_286 = arith.constant 286 : index
        %bidx_286 = arith.constant 537 : index
        %smidx_286 = arith.constant 957 : index
        %ssidx_286 = arith.constant 1377 : index
        %bias_286 = memref.load %in_v[%bidx_286] : memref<1511xi32>
        %smean_286 = memref.load %in_v[%smidx_286] : memref<1511xi32>
        %sscale_286 = memref.load %in_v[%ssidx_286] : memref<1511xi32>
        %dilation_286 = arith.constant 3 : i32
        %padding_286 = arith.constant 12 : i32
        %ki0_286 = arith.constant 1 : i32
        %ki1_286 = arith.constant 3 : i32
        %ki2_286 = arith.constant 4 : i32
        %pad1_286 = arith.constant 0 : i32
        %outidx_286 = arith.constant 286 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_286, %smean_286, %sscale_286, %dilation_286, %padding_286, %ki0_286, %ki1_286, %ki2_286, %pad1_286, %out_v, %outidx_286) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 287: dilation=3 padding=12 kernel=(1,3,5) padding1=1
        %fidx_287 = arith.constant 287 : index
        %bidx_287 = arith.constant 538 : index
        %smidx_287 = arith.constant 958 : index
        %ssidx_287 = arith.constant 1378 : index
        %bias_287 = memref.load %in_v[%bidx_287] : memref<1511xi32>
        %smean_287 = memref.load %in_v[%smidx_287] : memref<1511xi32>
        %sscale_287 = memref.load %in_v[%ssidx_287] : memref<1511xi32>
        %dilation_287 = arith.constant 3 : i32
        %padding_287 = arith.constant 12 : i32
        %ki0_287 = arith.constant 1 : i32
        %ki1_287 = arith.constant 3 : i32
        %ki2_287 = arith.constant 5 : i32
        %pad1_287 = arith.constant 1 : i32
        %outidx_287 = arith.constant 287 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_287, %smean_287, %sscale_287, %dilation_287, %padding_287, %ki0_287, %ki1_287, %ki2_287, %pad1_287, %out_v, %outidx_287) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 288: dilation=3 padding=12 kernel=(1,3,6) padding1=0
        %fidx_288 = arith.constant 288 : index
        %bidx_288 = arith.constant 539 : index
        %smidx_288 = arith.constant 959 : index
        %ssidx_288 = arith.constant 1379 : index
        %bias_288 = memref.load %in_v[%bidx_288] : memref<1511xi32>
        %smean_288 = memref.load %in_v[%smidx_288] : memref<1511xi32>
        %sscale_288 = memref.load %in_v[%ssidx_288] : memref<1511xi32>
        %dilation_288 = arith.constant 3 : i32
        %padding_288 = arith.constant 12 : i32
        %ki0_288 = arith.constant 1 : i32
        %ki1_288 = arith.constant 3 : i32
        %ki2_288 = arith.constant 6 : i32
        %pad1_288 = arith.constant 0 : i32
        %outidx_288 = arith.constant 288 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_288, %smean_288, %sscale_288, %dilation_288, %padding_288, %ki0_288, %ki1_288, %ki2_288, %pad1_288, %out_v, %outidx_288) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 289: dilation=3 padding=12 kernel=(1,3,7) padding1=1
        %fidx_289 = arith.constant 289 : index
        %bidx_289 = arith.constant 540 : index
        %smidx_289 = arith.constant 960 : index
        %ssidx_289 = arith.constant 1380 : index
        %bias_289 = memref.load %in_v[%bidx_289] : memref<1511xi32>
        %smean_289 = memref.load %in_v[%smidx_289] : memref<1511xi32>
        %sscale_289 = memref.load %in_v[%ssidx_289] : memref<1511xi32>
        %dilation_289 = arith.constant 3 : i32
        %padding_289 = arith.constant 12 : i32
        %ki0_289 = arith.constant 1 : i32
        %ki1_289 = arith.constant 3 : i32
        %ki2_289 = arith.constant 7 : i32
        %pad1_289 = arith.constant 1 : i32
        %outidx_289 = arith.constant 289 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_289, %smean_289, %sscale_289, %dilation_289, %padding_289, %ki0_289, %ki1_289, %ki2_289, %pad1_289, %out_v, %outidx_289) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 290: dilation=3 padding=12 kernel=(1,3,8) padding1=0
        %fidx_290 = arith.constant 290 : index
        %bidx_290 = arith.constant 541 : index
        %smidx_290 = arith.constant 961 : index
        %ssidx_290 = arith.constant 1381 : index
        %bias_290 = memref.load %in_v[%bidx_290] : memref<1511xi32>
        %smean_290 = memref.load %in_v[%smidx_290] : memref<1511xi32>
        %sscale_290 = memref.load %in_v[%ssidx_290] : memref<1511xi32>
        %dilation_290 = arith.constant 3 : i32
        %padding_290 = arith.constant 12 : i32
        %ki0_290 = arith.constant 1 : i32
        %ki1_290 = arith.constant 3 : i32
        %ki2_290 = arith.constant 8 : i32
        %pad1_290 = arith.constant 0 : i32
        %outidx_290 = arith.constant 290 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_290, %smean_290, %sscale_290, %dilation_290, %padding_290, %ki0_290, %ki1_290, %ki2_290, %pad1_290, %out_v, %outidx_290) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 291: dilation=3 padding=12 kernel=(1,4,5) padding1=1
        %fidx_291 = arith.constant 291 : index
        %bidx_291 = arith.constant 542 : index
        %smidx_291 = arith.constant 962 : index
        %ssidx_291 = arith.constant 1382 : index
        %bias_291 = memref.load %in_v[%bidx_291] : memref<1511xi32>
        %smean_291 = memref.load %in_v[%smidx_291] : memref<1511xi32>
        %sscale_291 = memref.load %in_v[%ssidx_291] : memref<1511xi32>
        %dilation_291 = arith.constant 3 : i32
        %padding_291 = arith.constant 12 : i32
        %ki0_291 = arith.constant 1 : i32
        %ki1_291 = arith.constant 4 : i32
        %ki2_291 = arith.constant 5 : i32
        %pad1_291 = arith.constant 1 : i32
        %outidx_291 = arith.constant 291 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_291, %smean_291, %sscale_291, %dilation_291, %padding_291, %ki0_291, %ki1_291, %ki2_291, %pad1_291, %out_v, %outidx_291) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 292: dilation=3 padding=12 kernel=(1,4,6) padding1=0
        %fidx_292 = arith.constant 292 : index
        %bidx_292 = arith.constant 543 : index
        %smidx_292 = arith.constant 963 : index
        %ssidx_292 = arith.constant 1383 : index
        %bias_292 = memref.load %in_v[%bidx_292] : memref<1511xi32>
        %smean_292 = memref.load %in_v[%smidx_292] : memref<1511xi32>
        %sscale_292 = memref.load %in_v[%ssidx_292] : memref<1511xi32>
        %dilation_292 = arith.constant 3 : i32
        %padding_292 = arith.constant 12 : i32
        %ki0_292 = arith.constant 1 : i32
        %ki1_292 = arith.constant 4 : i32
        %ki2_292 = arith.constant 6 : i32
        %pad1_292 = arith.constant 0 : i32
        %outidx_292 = arith.constant 292 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_292, %smean_292, %sscale_292, %dilation_292, %padding_292, %ki0_292, %ki1_292, %ki2_292, %pad1_292, %out_v, %outidx_292) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 293: dilation=3 padding=12 kernel=(1,4,7) padding1=1
        %fidx_293 = arith.constant 293 : index
        %bidx_293 = arith.constant 544 : index
        %smidx_293 = arith.constant 964 : index
        %ssidx_293 = arith.constant 1384 : index
        %bias_293 = memref.load %in_v[%bidx_293] : memref<1511xi32>
        %smean_293 = memref.load %in_v[%smidx_293] : memref<1511xi32>
        %sscale_293 = memref.load %in_v[%ssidx_293] : memref<1511xi32>
        %dilation_293 = arith.constant 3 : i32
        %padding_293 = arith.constant 12 : i32
        %ki0_293 = arith.constant 1 : i32
        %ki1_293 = arith.constant 4 : i32
        %ki2_293 = arith.constant 7 : i32
        %pad1_293 = arith.constant 1 : i32
        %outidx_293 = arith.constant 293 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_293, %smean_293, %sscale_293, %dilation_293, %padding_293, %ki0_293, %ki1_293, %ki2_293, %pad1_293, %out_v, %outidx_293) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 294: dilation=3 padding=12 kernel=(1,4,8) padding1=0
        %fidx_294 = arith.constant 294 : index
        %bidx_294 = arith.constant 545 : index
        %smidx_294 = arith.constant 965 : index
        %ssidx_294 = arith.constant 1385 : index
        %bias_294 = memref.load %in_v[%bidx_294] : memref<1511xi32>
        %smean_294 = memref.load %in_v[%smidx_294] : memref<1511xi32>
        %sscale_294 = memref.load %in_v[%ssidx_294] : memref<1511xi32>
        %dilation_294 = arith.constant 3 : i32
        %padding_294 = arith.constant 12 : i32
        %ki0_294 = arith.constant 1 : i32
        %ki1_294 = arith.constant 4 : i32
        %ki2_294 = arith.constant 8 : i32
        %pad1_294 = arith.constant 0 : i32
        %outidx_294 = arith.constant 294 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_294, %smean_294, %sscale_294, %dilation_294, %padding_294, %ki0_294, %ki1_294, %ki2_294, %pad1_294, %out_v, %outidx_294) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 295: dilation=3 padding=12 kernel=(1,5,6) padding1=1
        %fidx_295 = arith.constant 295 : index
        %bidx_295 = arith.constant 546 : index
        %smidx_295 = arith.constant 966 : index
        %ssidx_295 = arith.constant 1386 : index
        %bias_295 = memref.load %in_v[%bidx_295] : memref<1511xi32>
        %smean_295 = memref.load %in_v[%smidx_295] : memref<1511xi32>
        %sscale_295 = memref.load %in_v[%ssidx_295] : memref<1511xi32>
        %dilation_295 = arith.constant 3 : i32
        %padding_295 = arith.constant 12 : i32
        %ki0_295 = arith.constant 1 : i32
        %ki1_295 = arith.constant 5 : i32
        %ki2_295 = arith.constant 6 : i32
        %pad1_295 = arith.constant 1 : i32
        %outidx_295 = arith.constant 295 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_295, %smean_295, %sscale_295, %dilation_295, %padding_295, %ki0_295, %ki1_295, %ki2_295, %pad1_295, %out_v, %outidx_295) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 296: dilation=3 padding=12 kernel=(1,5,7) padding1=0
        %fidx_296 = arith.constant 296 : index
        %bidx_296 = arith.constant 547 : index
        %smidx_296 = arith.constant 967 : index
        %ssidx_296 = arith.constant 1387 : index
        %bias_296 = memref.load %in_v[%bidx_296] : memref<1511xi32>
        %smean_296 = memref.load %in_v[%smidx_296] : memref<1511xi32>
        %sscale_296 = memref.load %in_v[%ssidx_296] : memref<1511xi32>
        %dilation_296 = arith.constant 3 : i32
        %padding_296 = arith.constant 12 : i32
        %ki0_296 = arith.constant 1 : i32
        %ki1_296 = arith.constant 5 : i32
        %ki2_296 = arith.constant 7 : i32
        %pad1_296 = arith.constant 0 : i32
        %outidx_296 = arith.constant 296 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_296, %smean_296, %sscale_296, %dilation_296, %padding_296, %ki0_296, %ki1_296, %ki2_296, %pad1_296, %out_v, %outidx_296) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 297: dilation=3 padding=12 kernel=(1,5,8) padding1=1
        %fidx_297 = arith.constant 297 : index
        %bidx_297 = arith.constant 548 : index
        %smidx_297 = arith.constant 968 : index
        %ssidx_297 = arith.constant 1388 : index
        %bias_297 = memref.load %in_v[%bidx_297] : memref<1511xi32>
        %smean_297 = memref.load %in_v[%smidx_297] : memref<1511xi32>
        %sscale_297 = memref.load %in_v[%ssidx_297] : memref<1511xi32>
        %dilation_297 = arith.constant 3 : i32
        %padding_297 = arith.constant 12 : i32
        %ki0_297 = arith.constant 1 : i32
        %ki1_297 = arith.constant 5 : i32
        %ki2_297 = arith.constant 8 : i32
        %pad1_297 = arith.constant 1 : i32
        %outidx_297 = arith.constant 297 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_297, %smean_297, %sscale_297, %dilation_297, %padding_297, %ki0_297, %ki1_297, %ki2_297, %pad1_297, %out_v, %outidx_297) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 298: dilation=3 padding=12 kernel=(1,6,7) padding1=0
        %fidx_298 = arith.constant 298 : index
        %bidx_298 = arith.constant 549 : index
        %smidx_298 = arith.constant 969 : index
        %ssidx_298 = arith.constant 1389 : index
        %bias_298 = memref.load %in_v[%bidx_298] : memref<1511xi32>
        %smean_298 = memref.load %in_v[%smidx_298] : memref<1511xi32>
        %sscale_298 = memref.load %in_v[%ssidx_298] : memref<1511xi32>
        %dilation_298 = arith.constant 3 : i32
        %padding_298 = arith.constant 12 : i32
        %ki0_298 = arith.constant 1 : i32
        %ki1_298 = arith.constant 6 : i32
        %ki2_298 = arith.constant 7 : i32
        %pad1_298 = arith.constant 0 : i32
        %outidx_298 = arith.constant 298 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_298, %smean_298, %sscale_298, %dilation_298, %padding_298, %ki0_298, %ki1_298, %ki2_298, %pad1_298, %out_v, %outidx_298) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 299: dilation=3 padding=12 kernel=(1,6,8) padding1=1
        %fidx_299 = arith.constant 299 : index
        %bidx_299 = arith.constant 550 : index
        %smidx_299 = arith.constant 970 : index
        %ssidx_299 = arith.constant 1390 : index
        %bias_299 = memref.load %in_v[%bidx_299] : memref<1511xi32>
        %smean_299 = memref.load %in_v[%smidx_299] : memref<1511xi32>
        %sscale_299 = memref.load %in_v[%ssidx_299] : memref<1511xi32>
        %dilation_299 = arith.constant 3 : i32
        %padding_299 = arith.constant 12 : i32
        %ki0_299 = arith.constant 1 : i32
        %ki1_299 = arith.constant 6 : i32
        %ki2_299 = arith.constant 8 : i32
        %pad1_299 = arith.constant 1 : i32
        %outidx_299 = arith.constant 299 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_299, %smean_299, %sscale_299, %dilation_299, %padding_299, %ki0_299, %ki1_299, %ki2_299, %pad1_299, %out_v, %outidx_299) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 300: dilation=3 padding=12 kernel=(1,7,8) padding1=0
        %fidx_300 = arith.constant 300 : index
        %bidx_300 = arith.constant 551 : index
        %smidx_300 = arith.constant 971 : index
        %ssidx_300 = arith.constant 1391 : index
        %bias_300 = memref.load %in_v[%bidx_300] : memref<1511xi32>
        %smean_300 = memref.load %in_v[%smidx_300] : memref<1511xi32>
        %sscale_300 = memref.load %in_v[%ssidx_300] : memref<1511xi32>
        %dilation_300 = arith.constant 3 : i32
        %padding_300 = arith.constant 12 : i32
        %ki0_300 = arith.constant 1 : i32
        %ki1_300 = arith.constant 7 : i32
        %ki2_300 = arith.constant 8 : i32
        %pad1_300 = arith.constant 0 : i32
        %outidx_300 = arith.constant 300 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_300, %smean_300, %sscale_300, %dilation_300, %padding_300, %ki0_300, %ki1_300, %ki2_300, %pad1_300, %out_v, %outidx_300) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 301: dilation=3 padding=12 kernel=(2,3,4) padding1=1
        %fidx_301 = arith.constant 301 : index
        %bidx_301 = arith.constant 552 : index
        %smidx_301 = arith.constant 972 : index
        %ssidx_301 = arith.constant 1392 : index
        %bias_301 = memref.load %in_v[%bidx_301] : memref<1511xi32>
        %smean_301 = memref.load %in_v[%smidx_301] : memref<1511xi32>
        %sscale_301 = memref.load %in_v[%ssidx_301] : memref<1511xi32>
        %dilation_301 = arith.constant 3 : i32
        %padding_301 = arith.constant 12 : i32
        %ki0_301 = arith.constant 2 : i32
        %ki1_301 = arith.constant 3 : i32
        %ki2_301 = arith.constant 4 : i32
        %pad1_301 = arith.constant 1 : i32
        %outidx_301 = arith.constant 301 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_301, %smean_301, %sscale_301, %dilation_301, %padding_301, %ki0_301, %ki1_301, %ki2_301, %pad1_301, %out_v, %outidx_301) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 302: dilation=3 padding=12 kernel=(2,3,5) padding1=0
        %fidx_302 = arith.constant 302 : index
        %bidx_302 = arith.constant 553 : index
        %smidx_302 = arith.constant 973 : index
        %ssidx_302 = arith.constant 1393 : index
        %bias_302 = memref.load %in_v[%bidx_302] : memref<1511xi32>
        %smean_302 = memref.load %in_v[%smidx_302] : memref<1511xi32>
        %sscale_302 = memref.load %in_v[%ssidx_302] : memref<1511xi32>
        %dilation_302 = arith.constant 3 : i32
        %padding_302 = arith.constant 12 : i32
        %ki0_302 = arith.constant 2 : i32
        %ki1_302 = arith.constant 3 : i32
        %ki2_302 = arith.constant 5 : i32
        %pad1_302 = arith.constant 0 : i32
        %outidx_302 = arith.constant 302 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_302, %smean_302, %sscale_302, %dilation_302, %padding_302, %ki0_302, %ki1_302, %ki2_302, %pad1_302, %out_v, %outidx_302) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 303: dilation=3 padding=12 kernel=(2,3,6) padding1=1
        %fidx_303 = arith.constant 303 : index
        %bidx_303 = arith.constant 554 : index
        %smidx_303 = arith.constant 974 : index
        %ssidx_303 = arith.constant 1394 : index
        %bias_303 = memref.load %in_v[%bidx_303] : memref<1511xi32>
        %smean_303 = memref.load %in_v[%smidx_303] : memref<1511xi32>
        %sscale_303 = memref.load %in_v[%ssidx_303] : memref<1511xi32>
        %dilation_303 = arith.constant 3 : i32
        %padding_303 = arith.constant 12 : i32
        %ki0_303 = arith.constant 2 : i32
        %ki1_303 = arith.constant 3 : i32
        %ki2_303 = arith.constant 6 : i32
        %pad1_303 = arith.constant 1 : i32
        %outidx_303 = arith.constant 303 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_303, %smean_303, %sscale_303, %dilation_303, %padding_303, %ki0_303, %ki1_303, %ki2_303, %pad1_303, %out_v, %outidx_303) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 304: dilation=3 padding=12 kernel=(2,3,7) padding1=0
        %fidx_304 = arith.constant 304 : index
        %bidx_304 = arith.constant 555 : index
        %smidx_304 = arith.constant 975 : index
        %ssidx_304 = arith.constant 1395 : index
        %bias_304 = memref.load %in_v[%bidx_304] : memref<1511xi32>
        %smean_304 = memref.load %in_v[%smidx_304] : memref<1511xi32>
        %sscale_304 = memref.load %in_v[%ssidx_304] : memref<1511xi32>
        %dilation_304 = arith.constant 3 : i32
        %padding_304 = arith.constant 12 : i32
        %ki0_304 = arith.constant 2 : i32
        %ki1_304 = arith.constant 3 : i32
        %ki2_304 = arith.constant 7 : i32
        %pad1_304 = arith.constant 0 : i32
        %outidx_304 = arith.constant 304 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_304, %smean_304, %sscale_304, %dilation_304, %padding_304, %ki0_304, %ki1_304, %ki2_304, %pad1_304, %out_v, %outidx_304) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 305: dilation=3 padding=12 kernel=(2,3,8) padding1=1
        %fidx_305 = arith.constant 305 : index
        %bidx_305 = arith.constant 556 : index
        %smidx_305 = arith.constant 976 : index
        %ssidx_305 = arith.constant 1396 : index
        %bias_305 = memref.load %in_v[%bidx_305] : memref<1511xi32>
        %smean_305 = memref.load %in_v[%smidx_305] : memref<1511xi32>
        %sscale_305 = memref.load %in_v[%ssidx_305] : memref<1511xi32>
        %dilation_305 = arith.constant 3 : i32
        %padding_305 = arith.constant 12 : i32
        %ki0_305 = arith.constant 2 : i32
        %ki1_305 = arith.constant 3 : i32
        %ki2_305 = arith.constant 8 : i32
        %pad1_305 = arith.constant 1 : i32
        %outidx_305 = arith.constant 305 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_305, %smean_305, %sscale_305, %dilation_305, %padding_305, %ki0_305, %ki1_305, %ki2_305, %pad1_305, %out_v, %outidx_305) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 306: dilation=3 padding=12 kernel=(2,4,5) padding1=0
        %fidx_306 = arith.constant 306 : index
        %bidx_306 = arith.constant 557 : index
        %smidx_306 = arith.constant 977 : index
        %ssidx_306 = arith.constant 1397 : index
        %bias_306 = memref.load %in_v[%bidx_306] : memref<1511xi32>
        %smean_306 = memref.load %in_v[%smidx_306] : memref<1511xi32>
        %sscale_306 = memref.load %in_v[%ssidx_306] : memref<1511xi32>
        %dilation_306 = arith.constant 3 : i32
        %padding_306 = arith.constant 12 : i32
        %ki0_306 = arith.constant 2 : i32
        %ki1_306 = arith.constant 4 : i32
        %ki2_306 = arith.constant 5 : i32
        %pad1_306 = arith.constant 0 : i32
        %outidx_306 = arith.constant 306 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_306, %smean_306, %sscale_306, %dilation_306, %padding_306, %ki0_306, %ki1_306, %ki2_306, %pad1_306, %out_v, %outidx_306) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 307: dilation=3 padding=12 kernel=(2,4,6) padding1=1
        %fidx_307 = arith.constant 307 : index
        %bidx_307 = arith.constant 558 : index
        %smidx_307 = arith.constant 978 : index
        %ssidx_307 = arith.constant 1398 : index
        %bias_307 = memref.load %in_v[%bidx_307] : memref<1511xi32>
        %smean_307 = memref.load %in_v[%smidx_307] : memref<1511xi32>
        %sscale_307 = memref.load %in_v[%ssidx_307] : memref<1511xi32>
        %dilation_307 = arith.constant 3 : i32
        %padding_307 = arith.constant 12 : i32
        %ki0_307 = arith.constant 2 : i32
        %ki1_307 = arith.constant 4 : i32
        %ki2_307 = arith.constant 6 : i32
        %pad1_307 = arith.constant 1 : i32
        %outidx_307 = arith.constant 307 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_307, %smean_307, %sscale_307, %dilation_307, %padding_307, %ki0_307, %ki1_307, %ki2_307, %pad1_307, %out_v, %outidx_307) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 308: dilation=3 padding=12 kernel=(2,4,7) padding1=0
        %fidx_308 = arith.constant 308 : index
        %bidx_308 = arith.constant 559 : index
        %smidx_308 = arith.constant 979 : index
        %ssidx_308 = arith.constant 1399 : index
        %bias_308 = memref.load %in_v[%bidx_308] : memref<1511xi32>
        %smean_308 = memref.load %in_v[%smidx_308] : memref<1511xi32>
        %sscale_308 = memref.load %in_v[%ssidx_308] : memref<1511xi32>
        %dilation_308 = arith.constant 3 : i32
        %padding_308 = arith.constant 12 : i32
        %ki0_308 = arith.constant 2 : i32
        %ki1_308 = arith.constant 4 : i32
        %ki2_308 = arith.constant 7 : i32
        %pad1_308 = arith.constant 0 : i32
        %outidx_308 = arith.constant 308 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_308, %smean_308, %sscale_308, %dilation_308, %padding_308, %ki0_308, %ki1_308, %ki2_308, %pad1_308, %out_v, %outidx_308) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 309: dilation=3 padding=12 kernel=(2,4,8) padding1=1
        %fidx_309 = arith.constant 309 : index
        %bidx_309 = arith.constant 560 : index
        %smidx_309 = arith.constant 980 : index
        %ssidx_309 = arith.constant 1400 : index
        %bias_309 = memref.load %in_v[%bidx_309] : memref<1511xi32>
        %smean_309 = memref.load %in_v[%smidx_309] : memref<1511xi32>
        %sscale_309 = memref.load %in_v[%ssidx_309] : memref<1511xi32>
        %dilation_309 = arith.constant 3 : i32
        %padding_309 = arith.constant 12 : i32
        %ki0_309 = arith.constant 2 : i32
        %ki1_309 = arith.constant 4 : i32
        %ki2_309 = arith.constant 8 : i32
        %pad1_309 = arith.constant 1 : i32
        %outidx_309 = arith.constant 309 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_309, %smean_309, %sscale_309, %dilation_309, %padding_309, %ki0_309, %ki1_309, %ki2_309, %pad1_309, %out_v, %outidx_309) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 310: dilation=3 padding=12 kernel=(2,5,6) padding1=0
        %fidx_310 = arith.constant 310 : index
        %bidx_310 = arith.constant 561 : index
        %smidx_310 = arith.constant 981 : index
        %ssidx_310 = arith.constant 1401 : index
        %bias_310 = memref.load %in_v[%bidx_310] : memref<1511xi32>
        %smean_310 = memref.load %in_v[%smidx_310] : memref<1511xi32>
        %sscale_310 = memref.load %in_v[%ssidx_310] : memref<1511xi32>
        %dilation_310 = arith.constant 3 : i32
        %padding_310 = arith.constant 12 : i32
        %ki0_310 = arith.constant 2 : i32
        %ki1_310 = arith.constant 5 : i32
        %ki2_310 = arith.constant 6 : i32
        %pad1_310 = arith.constant 0 : i32
        %outidx_310 = arith.constant 310 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_310, %smean_310, %sscale_310, %dilation_310, %padding_310, %ki0_310, %ki1_310, %ki2_310, %pad1_310, %out_v, %outidx_310) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 311: dilation=3 padding=12 kernel=(2,5,7) padding1=1
        %fidx_311 = arith.constant 311 : index
        %bidx_311 = arith.constant 562 : index
        %smidx_311 = arith.constant 982 : index
        %ssidx_311 = arith.constant 1402 : index
        %bias_311 = memref.load %in_v[%bidx_311] : memref<1511xi32>
        %smean_311 = memref.load %in_v[%smidx_311] : memref<1511xi32>
        %sscale_311 = memref.load %in_v[%ssidx_311] : memref<1511xi32>
        %dilation_311 = arith.constant 3 : i32
        %padding_311 = arith.constant 12 : i32
        %ki0_311 = arith.constant 2 : i32
        %ki1_311 = arith.constant 5 : i32
        %ki2_311 = arith.constant 7 : i32
        %pad1_311 = arith.constant 1 : i32
        %outidx_311 = arith.constant 311 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_311, %smean_311, %sscale_311, %dilation_311, %padding_311, %ki0_311, %ki1_311, %ki2_311, %pad1_311, %out_v, %outidx_311) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 312: dilation=3 padding=12 kernel=(2,5,8) padding1=0
        %fidx_312 = arith.constant 312 : index
        %bidx_312 = arith.constant 563 : index
        %smidx_312 = arith.constant 983 : index
        %ssidx_312 = arith.constant 1403 : index
        %bias_312 = memref.load %in_v[%bidx_312] : memref<1511xi32>
        %smean_312 = memref.load %in_v[%smidx_312] : memref<1511xi32>
        %sscale_312 = memref.load %in_v[%ssidx_312] : memref<1511xi32>
        %dilation_312 = arith.constant 3 : i32
        %padding_312 = arith.constant 12 : i32
        %ki0_312 = arith.constant 2 : i32
        %ki1_312 = arith.constant 5 : i32
        %ki2_312 = arith.constant 8 : i32
        %pad1_312 = arith.constant 0 : i32
        %outidx_312 = arith.constant 312 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_312, %smean_312, %sscale_312, %dilation_312, %padding_312, %ki0_312, %ki1_312, %ki2_312, %pad1_312, %out_v, %outidx_312) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 313: dilation=3 padding=12 kernel=(2,6,7) padding1=1
        %fidx_313 = arith.constant 313 : index
        %bidx_313 = arith.constant 564 : index
        %smidx_313 = arith.constant 984 : index
        %ssidx_313 = arith.constant 1404 : index
        %bias_313 = memref.load %in_v[%bidx_313] : memref<1511xi32>
        %smean_313 = memref.load %in_v[%smidx_313] : memref<1511xi32>
        %sscale_313 = memref.load %in_v[%ssidx_313] : memref<1511xi32>
        %dilation_313 = arith.constant 3 : i32
        %padding_313 = arith.constant 12 : i32
        %ki0_313 = arith.constant 2 : i32
        %ki1_313 = arith.constant 6 : i32
        %ki2_313 = arith.constant 7 : i32
        %pad1_313 = arith.constant 1 : i32
        %outidx_313 = arith.constant 313 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_313, %smean_313, %sscale_313, %dilation_313, %padding_313, %ki0_313, %ki1_313, %ki2_313, %pad1_313, %out_v, %outidx_313) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 314: dilation=3 padding=12 kernel=(2,6,8) padding1=0
        %fidx_314 = arith.constant 314 : index
        %bidx_314 = arith.constant 565 : index
        %smidx_314 = arith.constant 985 : index
        %ssidx_314 = arith.constant 1405 : index
        %bias_314 = memref.load %in_v[%bidx_314] : memref<1511xi32>
        %smean_314 = memref.load %in_v[%smidx_314] : memref<1511xi32>
        %sscale_314 = memref.load %in_v[%ssidx_314] : memref<1511xi32>
        %dilation_314 = arith.constant 3 : i32
        %padding_314 = arith.constant 12 : i32
        %ki0_314 = arith.constant 2 : i32
        %ki1_314 = arith.constant 6 : i32
        %ki2_314 = arith.constant 8 : i32
        %pad1_314 = arith.constant 0 : i32
        %outidx_314 = arith.constant 314 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_314, %smean_314, %sscale_314, %dilation_314, %padding_314, %ki0_314, %ki1_314, %ki2_314, %pad1_314, %out_v, %outidx_314) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 315: dilation=3 padding=12 kernel=(2,7,8) padding1=1
        %fidx_315 = arith.constant 315 : index
        %bidx_315 = arith.constant 566 : index
        %smidx_315 = arith.constant 986 : index
        %ssidx_315 = arith.constant 1406 : index
        %bias_315 = memref.load %in_v[%bidx_315] : memref<1511xi32>
        %smean_315 = memref.load %in_v[%smidx_315] : memref<1511xi32>
        %sscale_315 = memref.load %in_v[%ssidx_315] : memref<1511xi32>
        %dilation_315 = arith.constant 3 : i32
        %padding_315 = arith.constant 12 : i32
        %ki0_315 = arith.constant 2 : i32
        %ki1_315 = arith.constant 7 : i32
        %ki2_315 = arith.constant 8 : i32
        %pad1_315 = arith.constant 1 : i32
        %outidx_315 = arith.constant 315 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_315, %smean_315, %sscale_315, %dilation_315, %padding_315, %ki0_315, %ki1_315, %ki2_315, %pad1_315, %out_v, %outidx_315) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 316: dilation=3 padding=12 kernel=(3,4,5) padding1=0
        %fidx_316 = arith.constant 316 : index
        %bidx_316 = arith.constant 567 : index
        %smidx_316 = arith.constant 987 : index
        %ssidx_316 = arith.constant 1407 : index
        %bias_316 = memref.load %in_v[%bidx_316] : memref<1511xi32>
        %smean_316 = memref.load %in_v[%smidx_316] : memref<1511xi32>
        %sscale_316 = memref.load %in_v[%ssidx_316] : memref<1511xi32>
        %dilation_316 = arith.constant 3 : i32
        %padding_316 = arith.constant 12 : i32
        %ki0_316 = arith.constant 3 : i32
        %ki1_316 = arith.constant 4 : i32
        %ki2_316 = arith.constant 5 : i32
        %pad1_316 = arith.constant 0 : i32
        %outidx_316 = arith.constant 316 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_316, %smean_316, %sscale_316, %dilation_316, %padding_316, %ki0_316, %ki1_316, %ki2_316, %pad1_316, %out_v, %outidx_316) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 317: dilation=3 padding=12 kernel=(3,4,6) padding1=1
        %fidx_317 = arith.constant 317 : index
        %bidx_317 = arith.constant 568 : index
        %smidx_317 = arith.constant 988 : index
        %ssidx_317 = arith.constant 1408 : index
        %bias_317 = memref.load %in_v[%bidx_317] : memref<1511xi32>
        %smean_317 = memref.load %in_v[%smidx_317] : memref<1511xi32>
        %sscale_317 = memref.load %in_v[%ssidx_317] : memref<1511xi32>
        %dilation_317 = arith.constant 3 : i32
        %padding_317 = arith.constant 12 : i32
        %ki0_317 = arith.constant 3 : i32
        %ki1_317 = arith.constant 4 : i32
        %ki2_317 = arith.constant 6 : i32
        %pad1_317 = arith.constant 1 : i32
        %outidx_317 = arith.constant 317 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_317, %smean_317, %sscale_317, %dilation_317, %padding_317, %ki0_317, %ki1_317, %ki2_317, %pad1_317, %out_v, %outidx_317) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 318: dilation=3 padding=12 kernel=(3,4,7) padding1=0
        %fidx_318 = arith.constant 318 : index
        %bidx_318 = arith.constant 569 : index
        %smidx_318 = arith.constant 989 : index
        %ssidx_318 = arith.constant 1409 : index
        %bias_318 = memref.load %in_v[%bidx_318] : memref<1511xi32>
        %smean_318 = memref.load %in_v[%smidx_318] : memref<1511xi32>
        %sscale_318 = memref.load %in_v[%ssidx_318] : memref<1511xi32>
        %dilation_318 = arith.constant 3 : i32
        %padding_318 = arith.constant 12 : i32
        %ki0_318 = arith.constant 3 : i32
        %ki1_318 = arith.constant 4 : i32
        %ki2_318 = arith.constant 7 : i32
        %pad1_318 = arith.constant 0 : i32
        %outidx_318 = arith.constant 318 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_318, %smean_318, %sscale_318, %dilation_318, %padding_318, %ki0_318, %ki1_318, %ki2_318, %pad1_318, %out_v, %outidx_318) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 319: dilation=3 padding=12 kernel=(3,4,8) padding1=1
        %fidx_319 = arith.constant 319 : index
        %bidx_319 = arith.constant 570 : index
        %smidx_319 = arith.constant 990 : index
        %ssidx_319 = arith.constant 1410 : index
        %bias_319 = memref.load %in_v[%bidx_319] : memref<1511xi32>
        %smean_319 = memref.load %in_v[%smidx_319] : memref<1511xi32>
        %sscale_319 = memref.load %in_v[%ssidx_319] : memref<1511xi32>
        %dilation_319 = arith.constant 3 : i32
        %padding_319 = arith.constant 12 : i32
        %ki0_319 = arith.constant 3 : i32
        %ki1_319 = arith.constant 4 : i32
        %ki2_319 = arith.constant 8 : i32
        %pad1_319 = arith.constant 1 : i32
        %outidx_319 = arith.constant 319 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_319, %smean_319, %sscale_319, %dilation_319, %padding_319, %ki0_319, %ki1_319, %ki2_319, %pad1_319, %out_v, %outidx_319) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 320: dilation=3 padding=12 kernel=(3,5,6) padding1=0
        %fidx_320 = arith.constant 320 : index
        %bidx_320 = arith.constant 571 : index
        %smidx_320 = arith.constant 991 : index
        %ssidx_320 = arith.constant 1411 : index
        %bias_320 = memref.load %in_v[%bidx_320] : memref<1511xi32>
        %smean_320 = memref.load %in_v[%smidx_320] : memref<1511xi32>
        %sscale_320 = memref.load %in_v[%ssidx_320] : memref<1511xi32>
        %dilation_320 = arith.constant 3 : i32
        %padding_320 = arith.constant 12 : i32
        %ki0_320 = arith.constant 3 : i32
        %ki1_320 = arith.constant 5 : i32
        %ki2_320 = arith.constant 6 : i32
        %pad1_320 = arith.constant 0 : i32
        %outidx_320 = arith.constant 320 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_320, %smean_320, %sscale_320, %dilation_320, %padding_320, %ki0_320, %ki1_320, %ki2_320, %pad1_320, %out_v, %outidx_320) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 321: dilation=3 padding=12 kernel=(3,5,7) padding1=1
        %fidx_321 = arith.constant 321 : index
        %bidx_321 = arith.constant 572 : index
        %smidx_321 = arith.constant 992 : index
        %ssidx_321 = arith.constant 1412 : index
        %bias_321 = memref.load %in_v[%bidx_321] : memref<1511xi32>
        %smean_321 = memref.load %in_v[%smidx_321] : memref<1511xi32>
        %sscale_321 = memref.load %in_v[%ssidx_321] : memref<1511xi32>
        %dilation_321 = arith.constant 3 : i32
        %padding_321 = arith.constant 12 : i32
        %ki0_321 = arith.constant 3 : i32
        %ki1_321 = arith.constant 5 : i32
        %ki2_321 = arith.constant 7 : i32
        %pad1_321 = arith.constant 1 : i32
        %outidx_321 = arith.constant 321 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_321, %smean_321, %sscale_321, %dilation_321, %padding_321, %ki0_321, %ki1_321, %ki2_321, %pad1_321, %out_v, %outidx_321) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 322: dilation=3 padding=12 kernel=(3,5,8) padding1=0
        %fidx_322 = arith.constant 322 : index
        %bidx_322 = arith.constant 573 : index
        %smidx_322 = arith.constant 993 : index
        %ssidx_322 = arith.constant 1413 : index
        %bias_322 = memref.load %in_v[%bidx_322] : memref<1511xi32>
        %smean_322 = memref.load %in_v[%smidx_322] : memref<1511xi32>
        %sscale_322 = memref.load %in_v[%ssidx_322] : memref<1511xi32>
        %dilation_322 = arith.constant 3 : i32
        %padding_322 = arith.constant 12 : i32
        %ki0_322 = arith.constant 3 : i32
        %ki1_322 = arith.constant 5 : i32
        %ki2_322 = arith.constant 8 : i32
        %pad1_322 = arith.constant 0 : i32
        %outidx_322 = arith.constant 322 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_322, %smean_322, %sscale_322, %dilation_322, %padding_322, %ki0_322, %ki1_322, %ki2_322, %pad1_322, %out_v, %outidx_322) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 323: dilation=3 padding=12 kernel=(3,6,7) padding1=1
        %fidx_323 = arith.constant 323 : index
        %bidx_323 = arith.constant 574 : index
        %smidx_323 = arith.constant 994 : index
        %ssidx_323 = arith.constant 1414 : index
        %bias_323 = memref.load %in_v[%bidx_323] : memref<1511xi32>
        %smean_323 = memref.load %in_v[%smidx_323] : memref<1511xi32>
        %sscale_323 = memref.load %in_v[%ssidx_323] : memref<1511xi32>
        %dilation_323 = arith.constant 3 : i32
        %padding_323 = arith.constant 12 : i32
        %ki0_323 = arith.constant 3 : i32
        %ki1_323 = arith.constant 6 : i32
        %ki2_323 = arith.constant 7 : i32
        %pad1_323 = arith.constant 1 : i32
        %outidx_323 = arith.constant 323 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_323, %smean_323, %sscale_323, %dilation_323, %padding_323, %ki0_323, %ki1_323, %ki2_323, %pad1_323, %out_v, %outidx_323) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 324: dilation=3 padding=12 kernel=(3,6,8) padding1=0
        %fidx_324 = arith.constant 324 : index
        %bidx_324 = arith.constant 575 : index
        %smidx_324 = arith.constant 995 : index
        %ssidx_324 = arith.constant 1415 : index
        %bias_324 = memref.load %in_v[%bidx_324] : memref<1511xi32>
        %smean_324 = memref.load %in_v[%smidx_324] : memref<1511xi32>
        %sscale_324 = memref.load %in_v[%ssidx_324] : memref<1511xi32>
        %dilation_324 = arith.constant 3 : i32
        %padding_324 = arith.constant 12 : i32
        %ki0_324 = arith.constant 3 : i32
        %ki1_324 = arith.constant 6 : i32
        %ki2_324 = arith.constant 8 : i32
        %pad1_324 = arith.constant 0 : i32
        %outidx_324 = arith.constant 324 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_324, %smean_324, %sscale_324, %dilation_324, %padding_324, %ki0_324, %ki1_324, %ki2_324, %pad1_324, %out_v, %outidx_324) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 325: dilation=3 padding=12 kernel=(3,7,8) padding1=1
        %fidx_325 = arith.constant 325 : index
        %bidx_325 = arith.constant 576 : index
        %smidx_325 = arith.constant 996 : index
        %ssidx_325 = arith.constant 1416 : index
        %bias_325 = memref.load %in_v[%bidx_325] : memref<1511xi32>
        %smean_325 = memref.load %in_v[%smidx_325] : memref<1511xi32>
        %sscale_325 = memref.load %in_v[%ssidx_325] : memref<1511xi32>
        %dilation_325 = arith.constant 3 : i32
        %padding_325 = arith.constant 12 : i32
        %ki0_325 = arith.constant 3 : i32
        %ki1_325 = arith.constant 7 : i32
        %ki2_325 = arith.constant 8 : i32
        %pad1_325 = arith.constant 1 : i32
        %outidx_325 = arith.constant 325 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_325, %smean_325, %sscale_325, %dilation_325, %padding_325, %ki0_325, %ki1_325, %ki2_325, %pad1_325, %out_v, %outidx_325) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 326: dilation=3 padding=12 kernel=(4,5,6) padding1=0
        %fidx_326 = arith.constant 326 : index
        %bidx_326 = arith.constant 577 : index
        %smidx_326 = arith.constant 997 : index
        %ssidx_326 = arith.constant 1417 : index
        %bias_326 = memref.load %in_v[%bidx_326] : memref<1511xi32>
        %smean_326 = memref.load %in_v[%smidx_326] : memref<1511xi32>
        %sscale_326 = memref.load %in_v[%ssidx_326] : memref<1511xi32>
        %dilation_326 = arith.constant 3 : i32
        %padding_326 = arith.constant 12 : i32
        %ki0_326 = arith.constant 4 : i32
        %ki1_326 = arith.constant 5 : i32
        %ki2_326 = arith.constant 6 : i32
        %pad1_326 = arith.constant 0 : i32
        %outidx_326 = arith.constant 326 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_326, %smean_326, %sscale_326, %dilation_326, %padding_326, %ki0_326, %ki1_326, %ki2_326, %pad1_326, %out_v, %outidx_326) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 327: dilation=3 padding=12 kernel=(4,5,7) padding1=1
        %fidx_327 = arith.constant 327 : index
        %bidx_327 = arith.constant 578 : index
        %smidx_327 = arith.constant 998 : index
        %ssidx_327 = arith.constant 1418 : index
        %bias_327 = memref.load %in_v[%bidx_327] : memref<1511xi32>
        %smean_327 = memref.load %in_v[%smidx_327] : memref<1511xi32>
        %sscale_327 = memref.load %in_v[%ssidx_327] : memref<1511xi32>
        %dilation_327 = arith.constant 3 : i32
        %padding_327 = arith.constant 12 : i32
        %ki0_327 = arith.constant 4 : i32
        %ki1_327 = arith.constant 5 : i32
        %ki2_327 = arith.constant 7 : i32
        %pad1_327 = arith.constant 1 : i32
        %outidx_327 = arith.constant 327 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_327, %smean_327, %sscale_327, %dilation_327, %padding_327, %ki0_327, %ki1_327, %ki2_327, %pad1_327, %out_v, %outidx_327) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 328: dilation=3 padding=12 kernel=(4,5,8) padding1=0
        %fidx_328 = arith.constant 328 : index
        %bidx_328 = arith.constant 579 : index
        %smidx_328 = arith.constant 999 : index
        %ssidx_328 = arith.constant 1419 : index
        %bias_328 = memref.load %in_v[%bidx_328] : memref<1511xi32>
        %smean_328 = memref.load %in_v[%smidx_328] : memref<1511xi32>
        %sscale_328 = memref.load %in_v[%ssidx_328] : memref<1511xi32>
        %dilation_328 = arith.constant 3 : i32
        %padding_328 = arith.constant 12 : i32
        %ki0_328 = arith.constant 4 : i32
        %ki1_328 = arith.constant 5 : i32
        %ki2_328 = arith.constant 8 : i32
        %pad1_328 = arith.constant 0 : i32
        %outidx_328 = arith.constant 328 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_328, %smean_328, %sscale_328, %dilation_328, %padding_328, %ki0_328, %ki1_328, %ki2_328, %pad1_328, %out_v, %outidx_328) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 329: dilation=3 padding=12 kernel=(4,6,7) padding1=1
        %fidx_329 = arith.constant 329 : index
        %bidx_329 = arith.constant 580 : index
        %smidx_329 = arith.constant 1000 : index
        %ssidx_329 = arith.constant 1420 : index
        %bias_329 = memref.load %in_v[%bidx_329] : memref<1511xi32>
        %smean_329 = memref.load %in_v[%smidx_329] : memref<1511xi32>
        %sscale_329 = memref.load %in_v[%ssidx_329] : memref<1511xi32>
        %dilation_329 = arith.constant 3 : i32
        %padding_329 = arith.constant 12 : i32
        %ki0_329 = arith.constant 4 : i32
        %ki1_329 = arith.constant 6 : i32
        %ki2_329 = arith.constant 7 : i32
        %pad1_329 = arith.constant 1 : i32
        %outidx_329 = arith.constant 329 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_329, %smean_329, %sscale_329, %dilation_329, %padding_329, %ki0_329, %ki1_329, %ki2_329, %pad1_329, %out_v, %outidx_329) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 330: dilation=3 padding=12 kernel=(4,6,8) padding1=0
        %fidx_330 = arith.constant 330 : index
        %bidx_330 = arith.constant 581 : index
        %smidx_330 = arith.constant 1001 : index
        %ssidx_330 = arith.constant 1421 : index
        %bias_330 = memref.load %in_v[%bidx_330] : memref<1511xi32>
        %smean_330 = memref.load %in_v[%smidx_330] : memref<1511xi32>
        %sscale_330 = memref.load %in_v[%ssidx_330] : memref<1511xi32>
        %dilation_330 = arith.constant 3 : i32
        %padding_330 = arith.constant 12 : i32
        %ki0_330 = arith.constant 4 : i32
        %ki1_330 = arith.constant 6 : i32
        %ki2_330 = arith.constant 8 : i32
        %pad1_330 = arith.constant 0 : i32
        %outidx_330 = arith.constant 330 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_330, %smean_330, %sscale_330, %dilation_330, %padding_330, %ki0_330, %ki1_330, %ki2_330, %pad1_330, %out_v, %outidx_330) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 331: dilation=3 padding=12 kernel=(4,7,8) padding1=1
        %fidx_331 = arith.constant 331 : index
        %bidx_331 = arith.constant 582 : index
        %smidx_331 = arith.constant 1002 : index
        %ssidx_331 = arith.constant 1422 : index
        %bias_331 = memref.load %in_v[%bidx_331] : memref<1511xi32>
        %smean_331 = memref.load %in_v[%smidx_331] : memref<1511xi32>
        %sscale_331 = memref.load %in_v[%ssidx_331] : memref<1511xi32>
        %dilation_331 = arith.constant 3 : i32
        %padding_331 = arith.constant 12 : i32
        %ki0_331 = arith.constant 4 : i32
        %ki1_331 = arith.constant 7 : i32
        %ki2_331 = arith.constant 8 : i32
        %pad1_331 = arith.constant 1 : i32
        %outidx_331 = arith.constant 331 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_331, %smean_331, %sscale_331, %dilation_331, %padding_331, %ki0_331, %ki1_331, %ki2_331, %pad1_331, %out_v, %outidx_331) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 332: dilation=3 padding=12 kernel=(5,6,7) padding1=0
        %fidx_332 = arith.constant 332 : index
        %bidx_332 = arith.constant 583 : index
        %smidx_332 = arith.constant 1003 : index
        %ssidx_332 = arith.constant 1423 : index
        %bias_332 = memref.load %in_v[%bidx_332] : memref<1511xi32>
        %smean_332 = memref.load %in_v[%smidx_332] : memref<1511xi32>
        %sscale_332 = memref.load %in_v[%ssidx_332] : memref<1511xi32>
        %dilation_332 = arith.constant 3 : i32
        %padding_332 = arith.constant 12 : i32
        %ki0_332 = arith.constant 5 : i32
        %ki1_332 = arith.constant 6 : i32
        %ki2_332 = arith.constant 7 : i32
        %pad1_332 = arith.constant 0 : i32
        %outidx_332 = arith.constant 332 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_332, %smean_332, %sscale_332, %dilation_332, %padding_332, %ki0_332, %ki1_332, %ki2_332, %pad1_332, %out_v, %outidx_332) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 333: dilation=3 padding=12 kernel=(5,6,8) padding1=1
        %fidx_333 = arith.constant 333 : index
        %bidx_333 = arith.constant 584 : index
        %smidx_333 = arith.constant 1004 : index
        %ssidx_333 = arith.constant 1424 : index
        %bias_333 = memref.load %in_v[%bidx_333] : memref<1511xi32>
        %smean_333 = memref.load %in_v[%smidx_333] : memref<1511xi32>
        %sscale_333 = memref.load %in_v[%ssidx_333] : memref<1511xi32>
        %dilation_333 = arith.constant 3 : i32
        %padding_333 = arith.constant 12 : i32
        %ki0_333 = arith.constant 5 : i32
        %ki1_333 = arith.constant 6 : i32
        %ki2_333 = arith.constant 8 : i32
        %pad1_333 = arith.constant 1 : i32
        %outidx_333 = arith.constant 333 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_333, %smean_333, %sscale_333, %dilation_333, %padding_333, %ki0_333, %ki1_333, %ki2_333, %pad1_333, %out_v, %outidx_333) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 334: dilation=3 padding=12 kernel=(5,7,8) padding1=0
        %fidx_334 = arith.constant 334 : index
        %bidx_334 = arith.constant 585 : index
        %smidx_334 = arith.constant 1005 : index
        %ssidx_334 = arith.constant 1425 : index
        %bias_334 = memref.load %in_v[%bidx_334] : memref<1511xi32>
        %smean_334 = memref.load %in_v[%smidx_334] : memref<1511xi32>
        %sscale_334 = memref.load %in_v[%ssidx_334] : memref<1511xi32>
        %dilation_334 = arith.constant 3 : i32
        %padding_334 = arith.constant 12 : i32
        %ki0_334 = arith.constant 5 : i32
        %ki1_334 = arith.constant 7 : i32
        %ki2_334 = arith.constant 8 : i32
        %pad1_334 = arith.constant 0 : i32
        %outidx_334 = arith.constant 334 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_334, %smean_334, %sscale_334, %dilation_334, %padding_334, %ki0_334, %ki1_334, %ki2_334, %pad1_334, %out_v, %outidx_334) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 335: dilation=3 padding=12 kernel=(6,7,8) padding1=1
        %fidx_335 = arith.constant 335 : index
        %bidx_335 = arith.constant 586 : index
        %smidx_335 = arith.constant 1006 : index
        %ssidx_335 = arith.constant 1426 : index
        %bias_335 = memref.load %in_v[%bidx_335] : memref<1511xi32>
        %smean_335 = memref.load %in_v[%smidx_335] : memref<1511xi32>
        %sscale_335 = memref.load %in_v[%ssidx_335] : memref<1511xi32>
        %dilation_335 = arith.constant 3 : i32
        %padding_335 = arith.constant 12 : i32
        %ki0_335 = arith.constant 6 : i32
        %ki1_335 = arith.constant 7 : i32
        %ki2_335 = arith.constant 8 : i32
        %pad1_335 = arith.constant 1 : i32
        %outidx_335 = arith.constant 335 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_335, %smean_335, %sscale_335, %dilation_335, %padding_335, %ki0_335, %ki1_335, %ki2_335, %pad1_335, %out_v, %outidx_335) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 336: dilation=4 padding=16 kernel=(0,1,2) padding1=1
        %fidx_336 = arith.constant 336 : index
        %bidx_336 = arith.constant 587 : index
        %smidx_336 = arith.constant 1007 : index
        %ssidx_336 = arith.constant 1427 : index
        %bias_336 = memref.load %in_v[%bidx_336] : memref<1511xi32>
        %smean_336 = memref.load %in_v[%smidx_336] : memref<1511xi32>
        %sscale_336 = memref.load %in_v[%ssidx_336] : memref<1511xi32>
        %dilation_336 = arith.constant 4 : i32
        %padding_336 = arith.constant 16 : i32
        %ki0_336 = arith.constant 0 : i32
        %ki1_336 = arith.constant 1 : i32
        %ki2_336 = arith.constant 2 : i32
        %pad1_336 = arith.constant 1 : i32
        %outidx_336 = arith.constant 336 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_336, %smean_336, %sscale_336, %dilation_336, %padding_336, %ki0_336, %ki1_336, %ki2_336, %pad1_336, %out_v, %outidx_336) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 337: dilation=4 padding=16 kernel=(0,1,3) padding1=0
        %fidx_337 = arith.constant 337 : index
        %bidx_337 = arith.constant 588 : index
        %smidx_337 = arith.constant 1008 : index
        %ssidx_337 = arith.constant 1428 : index
        %bias_337 = memref.load %in_v[%bidx_337] : memref<1511xi32>
        %smean_337 = memref.load %in_v[%smidx_337] : memref<1511xi32>
        %sscale_337 = memref.load %in_v[%ssidx_337] : memref<1511xi32>
        %dilation_337 = arith.constant 4 : i32
        %padding_337 = arith.constant 16 : i32
        %ki0_337 = arith.constant 0 : i32
        %ki1_337 = arith.constant 1 : i32
        %ki2_337 = arith.constant 3 : i32
        %pad1_337 = arith.constant 0 : i32
        %outidx_337 = arith.constant 337 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_337, %smean_337, %sscale_337, %dilation_337, %padding_337, %ki0_337, %ki1_337, %ki2_337, %pad1_337, %out_v, %outidx_337) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 338: dilation=4 padding=16 kernel=(0,1,4) padding1=1
        %fidx_338 = arith.constant 338 : index
        %bidx_338 = arith.constant 589 : index
        %smidx_338 = arith.constant 1009 : index
        %ssidx_338 = arith.constant 1429 : index
        %bias_338 = memref.load %in_v[%bidx_338] : memref<1511xi32>
        %smean_338 = memref.load %in_v[%smidx_338] : memref<1511xi32>
        %sscale_338 = memref.load %in_v[%ssidx_338] : memref<1511xi32>
        %dilation_338 = arith.constant 4 : i32
        %padding_338 = arith.constant 16 : i32
        %ki0_338 = arith.constant 0 : i32
        %ki1_338 = arith.constant 1 : i32
        %ki2_338 = arith.constant 4 : i32
        %pad1_338 = arith.constant 1 : i32
        %outidx_338 = arith.constant 338 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_338, %smean_338, %sscale_338, %dilation_338, %padding_338, %ki0_338, %ki1_338, %ki2_338, %pad1_338, %out_v, %outidx_338) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 339: dilation=4 padding=16 kernel=(0,1,5) padding1=0
        %fidx_339 = arith.constant 339 : index
        %bidx_339 = arith.constant 590 : index
        %smidx_339 = arith.constant 1010 : index
        %ssidx_339 = arith.constant 1430 : index
        %bias_339 = memref.load %in_v[%bidx_339] : memref<1511xi32>
        %smean_339 = memref.load %in_v[%smidx_339] : memref<1511xi32>
        %sscale_339 = memref.load %in_v[%ssidx_339] : memref<1511xi32>
        %dilation_339 = arith.constant 4 : i32
        %padding_339 = arith.constant 16 : i32
        %ki0_339 = arith.constant 0 : i32
        %ki1_339 = arith.constant 1 : i32
        %ki2_339 = arith.constant 5 : i32
        %pad1_339 = arith.constant 0 : i32
        %outidx_339 = arith.constant 339 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_339, %smean_339, %sscale_339, %dilation_339, %padding_339, %ki0_339, %ki1_339, %ki2_339, %pad1_339, %out_v, %outidx_339) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 340: dilation=4 padding=16 kernel=(0,1,6) padding1=1
        %fidx_340 = arith.constant 340 : index
        %bidx_340 = arith.constant 591 : index
        %smidx_340 = arith.constant 1011 : index
        %ssidx_340 = arith.constant 1431 : index
        %bias_340 = memref.load %in_v[%bidx_340] : memref<1511xi32>
        %smean_340 = memref.load %in_v[%smidx_340] : memref<1511xi32>
        %sscale_340 = memref.load %in_v[%ssidx_340] : memref<1511xi32>
        %dilation_340 = arith.constant 4 : i32
        %padding_340 = arith.constant 16 : i32
        %ki0_340 = arith.constant 0 : i32
        %ki1_340 = arith.constant 1 : i32
        %ki2_340 = arith.constant 6 : i32
        %pad1_340 = arith.constant 1 : i32
        %outidx_340 = arith.constant 340 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_340, %smean_340, %sscale_340, %dilation_340, %padding_340, %ki0_340, %ki1_340, %ki2_340, %pad1_340, %out_v, %outidx_340) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 341: dilation=4 padding=16 kernel=(0,1,7) padding1=0
        %fidx_341 = arith.constant 341 : index
        %bidx_341 = arith.constant 592 : index
        %smidx_341 = arith.constant 1012 : index
        %ssidx_341 = arith.constant 1432 : index
        %bias_341 = memref.load %in_v[%bidx_341] : memref<1511xi32>
        %smean_341 = memref.load %in_v[%smidx_341] : memref<1511xi32>
        %sscale_341 = memref.load %in_v[%ssidx_341] : memref<1511xi32>
        %dilation_341 = arith.constant 4 : i32
        %padding_341 = arith.constant 16 : i32
        %ki0_341 = arith.constant 0 : i32
        %ki1_341 = arith.constant 1 : i32
        %ki2_341 = arith.constant 7 : i32
        %pad1_341 = arith.constant 0 : i32
        %outidx_341 = arith.constant 341 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_341, %smean_341, %sscale_341, %dilation_341, %padding_341, %ki0_341, %ki1_341, %ki2_341, %pad1_341, %out_v, %outidx_341) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 342: dilation=4 padding=16 kernel=(0,1,8) padding1=1
        %fidx_342 = arith.constant 342 : index
        %bidx_342 = arith.constant 593 : index
        %smidx_342 = arith.constant 1013 : index
        %ssidx_342 = arith.constant 1433 : index
        %bias_342 = memref.load %in_v[%bidx_342] : memref<1511xi32>
        %smean_342 = memref.load %in_v[%smidx_342] : memref<1511xi32>
        %sscale_342 = memref.load %in_v[%ssidx_342] : memref<1511xi32>
        %dilation_342 = arith.constant 4 : i32
        %padding_342 = arith.constant 16 : i32
        %ki0_342 = arith.constant 0 : i32
        %ki1_342 = arith.constant 1 : i32
        %ki2_342 = arith.constant 8 : i32
        %pad1_342 = arith.constant 1 : i32
        %outidx_342 = arith.constant 342 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_342, %smean_342, %sscale_342, %dilation_342, %padding_342, %ki0_342, %ki1_342, %ki2_342, %pad1_342, %out_v, %outidx_342) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 343: dilation=4 padding=16 kernel=(0,2,3) padding1=0
        %fidx_343 = arith.constant 343 : index
        %bidx_343 = arith.constant 594 : index
        %smidx_343 = arith.constant 1014 : index
        %ssidx_343 = arith.constant 1434 : index
        %bias_343 = memref.load %in_v[%bidx_343] : memref<1511xi32>
        %smean_343 = memref.load %in_v[%smidx_343] : memref<1511xi32>
        %sscale_343 = memref.load %in_v[%ssidx_343] : memref<1511xi32>
        %dilation_343 = arith.constant 4 : i32
        %padding_343 = arith.constant 16 : i32
        %ki0_343 = arith.constant 0 : i32
        %ki1_343 = arith.constant 2 : i32
        %ki2_343 = arith.constant 3 : i32
        %pad1_343 = arith.constant 0 : i32
        %outidx_343 = arith.constant 343 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_343, %smean_343, %sscale_343, %dilation_343, %padding_343, %ki0_343, %ki1_343, %ki2_343, %pad1_343, %out_v, %outidx_343) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 344: dilation=4 padding=16 kernel=(0,2,4) padding1=1
        %fidx_344 = arith.constant 344 : index
        %bidx_344 = arith.constant 595 : index
        %smidx_344 = arith.constant 1015 : index
        %ssidx_344 = arith.constant 1435 : index
        %bias_344 = memref.load %in_v[%bidx_344] : memref<1511xi32>
        %smean_344 = memref.load %in_v[%smidx_344] : memref<1511xi32>
        %sscale_344 = memref.load %in_v[%ssidx_344] : memref<1511xi32>
        %dilation_344 = arith.constant 4 : i32
        %padding_344 = arith.constant 16 : i32
        %ki0_344 = arith.constant 0 : i32
        %ki1_344 = arith.constant 2 : i32
        %ki2_344 = arith.constant 4 : i32
        %pad1_344 = arith.constant 1 : i32
        %outidx_344 = arith.constant 344 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_344, %smean_344, %sscale_344, %dilation_344, %padding_344, %ki0_344, %ki1_344, %ki2_344, %pad1_344, %out_v, %outidx_344) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 345: dilation=4 padding=16 kernel=(0,2,5) padding1=0
        %fidx_345 = arith.constant 345 : index
        %bidx_345 = arith.constant 596 : index
        %smidx_345 = arith.constant 1016 : index
        %ssidx_345 = arith.constant 1436 : index
        %bias_345 = memref.load %in_v[%bidx_345] : memref<1511xi32>
        %smean_345 = memref.load %in_v[%smidx_345] : memref<1511xi32>
        %sscale_345 = memref.load %in_v[%ssidx_345] : memref<1511xi32>
        %dilation_345 = arith.constant 4 : i32
        %padding_345 = arith.constant 16 : i32
        %ki0_345 = arith.constant 0 : i32
        %ki1_345 = arith.constant 2 : i32
        %ki2_345 = arith.constant 5 : i32
        %pad1_345 = arith.constant 0 : i32
        %outidx_345 = arith.constant 345 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_345, %smean_345, %sscale_345, %dilation_345, %padding_345, %ki0_345, %ki1_345, %ki2_345, %pad1_345, %out_v, %outidx_345) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 346: dilation=4 padding=16 kernel=(0,2,6) padding1=1
        %fidx_346 = arith.constant 346 : index
        %bidx_346 = arith.constant 597 : index
        %smidx_346 = arith.constant 1017 : index
        %ssidx_346 = arith.constant 1437 : index
        %bias_346 = memref.load %in_v[%bidx_346] : memref<1511xi32>
        %smean_346 = memref.load %in_v[%smidx_346] : memref<1511xi32>
        %sscale_346 = memref.load %in_v[%ssidx_346] : memref<1511xi32>
        %dilation_346 = arith.constant 4 : i32
        %padding_346 = arith.constant 16 : i32
        %ki0_346 = arith.constant 0 : i32
        %ki1_346 = arith.constant 2 : i32
        %ki2_346 = arith.constant 6 : i32
        %pad1_346 = arith.constant 1 : i32
        %outidx_346 = arith.constant 346 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_346, %smean_346, %sscale_346, %dilation_346, %padding_346, %ki0_346, %ki1_346, %ki2_346, %pad1_346, %out_v, %outidx_346) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 347: dilation=4 padding=16 kernel=(0,2,7) padding1=0
        %fidx_347 = arith.constant 347 : index
        %bidx_347 = arith.constant 598 : index
        %smidx_347 = arith.constant 1018 : index
        %ssidx_347 = arith.constant 1438 : index
        %bias_347 = memref.load %in_v[%bidx_347] : memref<1511xi32>
        %smean_347 = memref.load %in_v[%smidx_347] : memref<1511xi32>
        %sscale_347 = memref.load %in_v[%ssidx_347] : memref<1511xi32>
        %dilation_347 = arith.constant 4 : i32
        %padding_347 = arith.constant 16 : i32
        %ki0_347 = arith.constant 0 : i32
        %ki1_347 = arith.constant 2 : i32
        %ki2_347 = arith.constant 7 : i32
        %pad1_347 = arith.constant 0 : i32
        %outidx_347 = arith.constant 347 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_347, %smean_347, %sscale_347, %dilation_347, %padding_347, %ki0_347, %ki1_347, %ki2_347, %pad1_347, %out_v, %outidx_347) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 348: dilation=4 padding=16 kernel=(0,2,8) padding1=1
        %fidx_348 = arith.constant 348 : index
        %bidx_348 = arith.constant 599 : index
        %smidx_348 = arith.constant 1019 : index
        %ssidx_348 = arith.constant 1439 : index
        %bias_348 = memref.load %in_v[%bidx_348] : memref<1511xi32>
        %smean_348 = memref.load %in_v[%smidx_348] : memref<1511xi32>
        %sscale_348 = memref.load %in_v[%ssidx_348] : memref<1511xi32>
        %dilation_348 = arith.constant 4 : i32
        %padding_348 = arith.constant 16 : i32
        %ki0_348 = arith.constant 0 : i32
        %ki1_348 = arith.constant 2 : i32
        %ki2_348 = arith.constant 8 : i32
        %pad1_348 = arith.constant 1 : i32
        %outidx_348 = arith.constant 348 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_348, %smean_348, %sscale_348, %dilation_348, %padding_348, %ki0_348, %ki1_348, %ki2_348, %pad1_348, %out_v, %outidx_348) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 349: dilation=4 padding=16 kernel=(0,3,4) padding1=0
        %fidx_349 = arith.constant 349 : index
        %bidx_349 = arith.constant 600 : index
        %smidx_349 = arith.constant 1020 : index
        %ssidx_349 = arith.constant 1440 : index
        %bias_349 = memref.load %in_v[%bidx_349] : memref<1511xi32>
        %smean_349 = memref.load %in_v[%smidx_349] : memref<1511xi32>
        %sscale_349 = memref.load %in_v[%ssidx_349] : memref<1511xi32>
        %dilation_349 = arith.constant 4 : i32
        %padding_349 = arith.constant 16 : i32
        %ki0_349 = arith.constant 0 : i32
        %ki1_349 = arith.constant 3 : i32
        %ki2_349 = arith.constant 4 : i32
        %pad1_349 = arith.constant 0 : i32
        %outidx_349 = arith.constant 349 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_349, %smean_349, %sscale_349, %dilation_349, %padding_349, %ki0_349, %ki1_349, %ki2_349, %pad1_349, %out_v, %outidx_349) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 350: dilation=4 padding=16 kernel=(0,3,5) padding1=1
        %fidx_350 = arith.constant 350 : index
        %bidx_350 = arith.constant 601 : index
        %smidx_350 = arith.constant 1021 : index
        %ssidx_350 = arith.constant 1441 : index
        %bias_350 = memref.load %in_v[%bidx_350] : memref<1511xi32>
        %smean_350 = memref.load %in_v[%smidx_350] : memref<1511xi32>
        %sscale_350 = memref.load %in_v[%ssidx_350] : memref<1511xi32>
        %dilation_350 = arith.constant 4 : i32
        %padding_350 = arith.constant 16 : i32
        %ki0_350 = arith.constant 0 : i32
        %ki1_350 = arith.constant 3 : i32
        %ki2_350 = arith.constant 5 : i32
        %pad1_350 = arith.constant 1 : i32
        %outidx_350 = arith.constant 350 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_350, %smean_350, %sscale_350, %dilation_350, %padding_350, %ki0_350, %ki1_350, %ki2_350, %pad1_350, %out_v, %outidx_350) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 351: dilation=4 padding=16 kernel=(0,3,6) padding1=0
        %fidx_351 = arith.constant 351 : index
        %bidx_351 = arith.constant 602 : index
        %smidx_351 = arith.constant 1022 : index
        %ssidx_351 = arith.constant 1442 : index
        %bias_351 = memref.load %in_v[%bidx_351] : memref<1511xi32>
        %smean_351 = memref.load %in_v[%smidx_351] : memref<1511xi32>
        %sscale_351 = memref.load %in_v[%ssidx_351] : memref<1511xi32>
        %dilation_351 = arith.constant 4 : i32
        %padding_351 = arith.constant 16 : i32
        %ki0_351 = arith.constant 0 : i32
        %ki1_351 = arith.constant 3 : i32
        %ki2_351 = arith.constant 6 : i32
        %pad1_351 = arith.constant 0 : i32
        %outidx_351 = arith.constant 351 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_351, %smean_351, %sscale_351, %dilation_351, %padding_351, %ki0_351, %ki1_351, %ki2_351, %pad1_351, %out_v, %outidx_351) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 352: dilation=4 padding=16 kernel=(0,3,7) padding1=1
        %fidx_352 = arith.constant 352 : index
        %bidx_352 = arith.constant 603 : index
        %smidx_352 = arith.constant 1023 : index
        %ssidx_352 = arith.constant 1443 : index
        %bias_352 = memref.load %in_v[%bidx_352] : memref<1511xi32>
        %smean_352 = memref.load %in_v[%smidx_352] : memref<1511xi32>
        %sscale_352 = memref.load %in_v[%ssidx_352] : memref<1511xi32>
        %dilation_352 = arith.constant 4 : i32
        %padding_352 = arith.constant 16 : i32
        %ki0_352 = arith.constant 0 : i32
        %ki1_352 = arith.constant 3 : i32
        %ki2_352 = arith.constant 7 : i32
        %pad1_352 = arith.constant 1 : i32
        %outidx_352 = arith.constant 352 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_352, %smean_352, %sscale_352, %dilation_352, %padding_352, %ki0_352, %ki1_352, %ki2_352, %pad1_352, %out_v, %outidx_352) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 353: dilation=4 padding=16 kernel=(0,3,8) padding1=0
        %fidx_353 = arith.constant 353 : index
        %bidx_353 = arith.constant 604 : index
        %smidx_353 = arith.constant 1024 : index
        %ssidx_353 = arith.constant 1444 : index
        %bias_353 = memref.load %in_v[%bidx_353] : memref<1511xi32>
        %smean_353 = memref.load %in_v[%smidx_353] : memref<1511xi32>
        %sscale_353 = memref.load %in_v[%ssidx_353] : memref<1511xi32>
        %dilation_353 = arith.constant 4 : i32
        %padding_353 = arith.constant 16 : i32
        %ki0_353 = arith.constant 0 : i32
        %ki1_353 = arith.constant 3 : i32
        %ki2_353 = arith.constant 8 : i32
        %pad1_353 = arith.constant 0 : i32
        %outidx_353 = arith.constant 353 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_353, %smean_353, %sscale_353, %dilation_353, %padding_353, %ki0_353, %ki1_353, %ki2_353, %pad1_353, %out_v, %outidx_353) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 354: dilation=4 padding=16 kernel=(0,4,5) padding1=1
        %fidx_354 = arith.constant 354 : index
        %bidx_354 = arith.constant 605 : index
        %smidx_354 = arith.constant 1025 : index
        %ssidx_354 = arith.constant 1445 : index
        %bias_354 = memref.load %in_v[%bidx_354] : memref<1511xi32>
        %smean_354 = memref.load %in_v[%smidx_354] : memref<1511xi32>
        %sscale_354 = memref.load %in_v[%ssidx_354] : memref<1511xi32>
        %dilation_354 = arith.constant 4 : i32
        %padding_354 = arith.constant 16 : i32
        %ki0_354 = arith.constant 0 : i32
        %ki1_354 = arith.constant 4 : i32
        %ki2_354 = arith.constant 5 : i32
        %pad1_354 = arith.constant 1 : i32
        %outidx_354 = arith.constant 354 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_354, %smean_354, %sscale_354, %dilation_354, %padding_354, %ki0_354, %ki1_354, %ki2_354, %pad1_354, %out_v, %outidx_354) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 355: dilation=4 padding=16 kernel=(0,4,6) padding1=0
        %fidx_355 = arith.constant 355 : index
        %bidx_355 = arith.constant 606 : index
        %smidx_355 = arith.constant 1026 : index
        %ssidx_355 = arith.constant 1446 : index
        %bias_355 = memref.load %in_v[%bidx_355] : memref<1511xi32>
        %smean_355 = memref.load %in_v[%smidx_355] : memref<1511xi32>
        %sscale_355 = memref.load %in_v[%ssidx_355] : memref<1511xi32>
        %dilation_355 = arith.constant 4 : i32
        %padding_355 = arith.constant 16 : i32
        %ki0_355 = arith.constant 0 : i32
        %ki1_355 = arith.constant 4 : i32
        %ki2_355 = arith.constant 6 : i32
        %pad1_355 = arith.constant 0 : i32
        %outidx_355 = arith.constant 355 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_355, %smean_355, %sscale_355, %dilation_355, %padding_355, %ki0_355, %ki1_355, %ki2_355, %pad1_355, %out_v, %outidx_355) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 356: dilation=4 padding=16 kernel=(0,4,7) padding1=1
        %fidx_356 = arith.constant 356 : index
        %bidx_356 = arith.constant 607 : index
        %smidx_356 = arith.constant 1027 : index
        %ssidx_356 = arith.constant 1447 : index
        %bias_356 = memref.load %in_v[%bidx_356] : memref<1511xi32>
        %smean_356 = memref.load %in_v[%smidx_356] : memref<1511xi32>
        %sscale_356 = memref.load %in_v[%ssidx_356] : memref<1511xi32>
        %dilation_356 = arith.constant 4 : i32
        %padding_356 = arith.constant 16 : i32
        %ki0_356 = arith.constant 0 : i32
        %ki1_356 = arith.constant 4 : i32
        %ki2_356 = arith.constant 7 : i32
        %pad1_356 = arith.constant 1 : i32
        %outidx_356 = arith.constant 356 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_356, %smean_356, %sscale_356, %dilation_356, %padding_356, %ki0_356, %ki1_356, %ki2_356, %pad1_356, %out_v, %outidx_356) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 357: dilation=4 padding=16 kernel=(0,4,8) padding1=0
        %fidx_357 = arith.constant 357 : index
        %bidx_357 = arith.constant 608 : index
        %smidx_357 = arith.constant 1028 : index
        %ssidx_357 = arith.constant 1448 : index
        %bias_357 = memref.load %in_v[%bidx_357] : memref<1511xi32>
        %smean_357 = memref.load %in_v[%smidx_357] : memref<1511xi32>
        %sscale_357 = memref.load %in_v[%ssidx_357] : memref<1511xi32>
        %dilation_357 = arith.constant 4 : i32
        %padding_357 = arith.constant 16 : i32
        %ki0_357 = arith.constant 0 : i32
        %ki1_357 = arith.constant 4 : i32
        %ki2_357 = arith.constant 8 : i32
        %pad1_357 = arith.constant 0 : i32
        %outidx_357 = arith.constant 357 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_357, %smean_357, %sscale_357, %dilation_357, %padding_357, %ki0_357, %ki1_357, %ki2_357, %pad1_357, %out_v, %outidx_357) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 358: dilation=4 padding=16 kernel=(0,5,6) padding1=1
        %fidx_358 = arith.constant 358 : index
        %bidx_358 = arith.constant 609 : index
        %smidx_358 = arith.constant 1029 : index
        %ssidx_358 = arith.constant 1449 : index
        %bias_358 = memref.load %in_v[%bidx_358] : memref<1511xi32>
        %smean_358 = memref.load %in_v[%smidx_358] : memref<1511xi32>
        %sscale_358 = memref.load %in_v[%ssidx_358] : memref<1511xi32>
        %dilation_358 = arith.constant 4 : i32
        %padding_358 = arith.constant 16 : i32
        %ki0_358 = arith.constant 0 : i32
        %ki1_358 = arith.constant 5 : i32
        %ki2_358 = arith.constant 6 : i32
        %pad1_358 = arith.constant 1 : i32
        %outidx_358 = arith.constant 358 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_358, %smean_358, %sscale_358, %dilation_358, %padding_358, %ki0_358, %ki1_358, %ki2_358, %pad1_358, %out_v, %outidx_358) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 359: dilation=4 padding=16 kernel=(0,5,7) padding1=0
        %fidx_359 = arith.constant 359 : index
        %bidx_359 = arith.constant 610 : index
        %smidx_359 = arith.constant 1030 : index
        %ssidx_359 = arith.constant 1450 : index
        %bias_359 = memref.load %in_v[%bidx_359] : memref<1511xi32>
        %smean_359 = memref.load %in_v[%smidx_359] : memref<1511xi32>
        %sscale_359 = memref.load %in_v[%ssidx_359] : memref<1511xi32>
        %dilation_359 = arith.constant 4 : i32
        %padding_359 = arith.constant 16 : i32
        %ki0_359 = arith.constant 0 : i32
        %ki1_359 = arith.constant 5 : i32
        %ki2_359 = arith.constant 7 : i32
        %pad1_359 = arith.constant 0 : i32
        %outidx_359 = arith.constant 359 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_359, %smean_359, %sscale_359, %dilation_359, %padding_359, %ki0_359, %ki1_359, %ki2_359, %pad1_359, %out_v, %outidx_359) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 360: dilation=4 padding=16 kernel=(0,5,8) padding1=1
        %fidx_360 = arith.constant 360 : index
        %bidx_360 = arith.constant 611 : index
        %smidx_360 = arith.constant 1031 : index
        %ssidx_360 = arith.constant 1451 : index
        %bias_360 = memref.load %in_v[%bidx_360] : memref<1511xi32>
        %smean_360 = memref.load %in_v[%smidx_360] : memref<1511xi32>
        %sscale_360 = memref.load %in_v[%ssidx_360] : memref<1511xi32>
        %dilation_360 = arith.constant 4 : i32
        %padding_360 = arith.constant 16 : i32
        %ki0_360 = arith.constant 0 : i32
        %ki1_360 = arith.constant 5 : i32
        %ki2_360 = arith.constant 8 : i32
        %pad1_360 = arith.constant 1 : i32
        %outidx_360 = arith.constant 360 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_360, %smean_360, %sscale_360, %dilation_360, %padding_360, %ki0_360, %ki1_360, %ki2_360, %pad1_360, %out_v, %outidx_360) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 361: dilation=4 padding=16 kernel=(0,6,7) padding1=0
        %fidx_361 = arith.constant 361 : index
        %bidx_361 = arith.constant 612 : index
        %smidx_361 = arith.constant 1032 : index
        %ssidx_361 = arith.constant 1452 : index
        %bias_361 = memref.load %in_v[%bidx_361] : memref<1511xi32>
        %smean_361 = memref.load %in_v[%smidx_361] : memref<1511xi32>
        %sscale_361 = memref.load %in_v[%ssidx_361] : memref<1511xi32>
        %dilation_361 = arith.constant 4 : i32
        %padding_361 = arith.constant 16 : i32
        %ki0_361 = arith.constant 0 : i32
        %ki1_361 = arith.constant 6 : i32
        %ki2_361 = arith.constant 7 : i32
        %pad1_361 = arith.constant 0 : i32
        %outidx_361 = arith.constant 361 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_361, %smean_361, %sscale_361, %dilation_361, %padding_361, %ki0_361, %ki1_361, %ki2_361, %pad1_361, %out_v, %outidx_361) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 362: dilation=4 padding=16 kernel=(0,6,8) padding1=1
        %fidx_362 = arith.constant 362 : index
        %bidx_362 = arith.constant 613 : index
        %smidx_362 = arith.constant 1033 : index
        %ssidx_362 = arith.constant 1453 : index
        %bias_362 = memref.load %in_v[%bidx_362] : memref<1511xi32>
        %smean_362 = memref.load %in_v[%smidx_362] : memref<1511xi32>
        %sscale_362 = memref.load %in_v[%ssidx_362] : memref<1511xi32>
        %dilation_362 = arith.constant 4 : i32
        %padding_362 = arith.constant 16 : i32
        %ki0_362 = arith.constant 0 : i32
        %ki1_362 = arith.constant 6 : i32
        %ki2_362 = arith.constant 8 : i32
        %pad1_362 = arith.constant 1 : i32
        %outidx_362 = arith.constant 362 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_362, %smean_362, %sscale_362, %dilation_362, %padding_362, %ki0_362, %ki1_362, %ki2_362, %pad1_362, %out_v, %outidx_362) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 363: dilation=4 padding=16 kernel=(0,7,8) padding1=0
        %fidx_363 = arith.constant 363 : index
        %bidx_363 = arith.constant 614 : index
        %smidx_363 = arith.constant 1034 : index
        %ssidx_363 = arith.constant 1454 : index
        %bias_363 = memref.load %in_v[%bidx_363] : memref<1511xi32>
        %smean_363 = memref.load %in_v[%smidx_363] : memref<1511xi32>
        %sscale_363 = memref.load %in_v[%ssidx_363] : memref<1511xi32>
        %dilation_363 = arith.constant 4 : i32
        %padding_363 = arith.constant 16 : i32
        %ki0_363 = arith.constant 0 : i32
        %ki1_363 = arith.constant 7 : i32
        %ki2_363 = arith.constant 8 : i32
        %pad1_363 = arith.constant 0 : i32
        %outidx_363 = arith.constant 363 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_363, %smean_363, %sscale_363, %dilation_363, %padding_363, %ki0_363, %ki1_363, %ki2_363, %pad1_363, %out_v, %outidx_363) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 364: dilation=4 padding=16 kernel=(1,2,3) padding1=1
        %fidx_364 = arith.constant 364 : index
        %bidx_364 = arith.constant 615 : index
        %smidx_364 = arith.constant 1035 : index
        %ssidx_364 = arith.constant 1455 : index
        %bias_364 = memref.load %in_v[%bidx_364] : memref<1511xi32>
        %smean_364 = memref.load %in_v[%smidx_364] : memref<1511xi32>
        %sscale_364 = memref.load %in_v[%ssidx_364] : memref<1511xi32>
        %dilation_364 = arith.constant 4 : i32
        %padding_364 = arith.constant 16 : i32
        %ki0_364 = arith.constant 1 : i32
        %ki1_364 = arith.constant 2 : i32
        %ki2_364 = arith.constant 3 : i32
        %pad1_364 = arith.constant 1 : i32
        %outidx_364 = arith.constant 364 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_364, %smean_364, %sscale_364, %dilation_364, %padding_364, %ki0_364, %ki1_364, %ki2_364, %pad1_364, %out_v, %outidx_364) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 365: dilation=4 padding=16 kernel=(1,2,4) padding1=0
        %fidx_365 = arith.constant 365 : index
        %bidx_365 = arith.constant 616 : index
        %smidx_365 = arith.constant 1036 : index
        %ssidx_365 = arith.constant 1456 : index
        %bias_365 = memref.load %in_v[%bidx_365] : memref<1511xi32>
        %smean_365 = memref.load %in_v[%smidx_365] : memref<1511xi32>
        %sscale_365 = memref.load %in_v[%ssidx_365] : memref<1511xi32>
        %dilation_365 = arith.constant 4 : i32
        %padding_365 = arith.constant 16 : i32
        %ki0_365 = arith.constant 1 : i32
        %ki1_365 = arith.constant 2 : i32
        %ki2_365 = arith.constant 4 : i32
        %pad1_365 = arith.constant 0 : i32
        %outidx_365 = arith.constant 365 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_365, %smean_365, %sscale_365, %dilation_365, %padding_365, %ki0_365, %ki1_365, %ki2_365, %pad1_365, %out_v, %outidx_365) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 366: dilation=4 padding=16 kernel=(1,2,5) padding1=1
        %fidx_366 = arith.constant 366 : index
        %bidx_366 = arith.constant 617 : index
        %smidx_366 = arith.constant 1037 : index
        %ssidx_366 = arith.constant 1457 : index
        %bias_366 = memref.load %in_v[%bidx_366] : memref<1511xi32>
        %smean_366 = memref.load %in_v[%smidx_366] : memref<1511xi32>
        %sscale_366 = memref.load %in_v[%ssidx_366] : memref<1511xi32>
        %dilation_366 = arith.constant 4 : i32
        %padding_366 = arith.constant 16 : i32
        %ki0_366 = arith.constant 1 : i32
        %ki1_366 = arith.constant 2 : i32
        %ki2_366 = arith.constant 5 : i32
        %pad1_366 = arith.constant 1 : i32
        %outidx_366 = arith.constant 366 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_366, %smean_366, %sscale_366, %dilation_366, %padding_366, %ki0_366, %ki1_366, %ki2_366, %pad1_366, %out_v, %outidx_366) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 367: dilation=4 padding=16 kernel=(1,2,6) padding1=0
        %fidx_367 = arith.constant 367 : index
        %bidx_367 = arith.constant 618 : index
        %smidx_367 = arith.constant 1038 : index
        %ssidx_367 = arith.constant 1458 : index
        %bias_367 = memref.load %in_v[%bidx_367] : memref<1511xi32>
        %smean_367 = memref.load %in_v[%smidx_367] : memref<1511xi32>
        %sscale_367 = memref.load %in_v[%ssidx_367] : memref<1511xi32>
        %dilation_367 = arith.constant 4 : i32
        %padding_367 = arith.constant 16 : i32
        %ki0_367 = arith.constant 1 : i32
        %ki1_367 = arith.constant 2 : i32
        %ki2_367 = arith.constant 6 : i32
        %pad1_367 = arith.constant 0 : i32
        %outidx_367 = arith.constant 367 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_367, %smean_367, %sscale_367, %dilation_367, %padding_367, %ki0_367, %ki1_367, %ki2_367, %pad1_367, %out_v, %outidx_367) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 368: dilation=4 padding=16 kernel=(1,2,7) padding1=1
        %fidx_368 = arith.constant 368 : index
        %bidx_368 = arith.constant 619 : index
        %smidx_368 = arith.constant 1039 : index
        %ssidx_368 = arith.constant 1459 : index
        %bias_368 = memref.load %in_v[%bidx_368] : memref<1511xi32>
        %smean_368 = memref.load %in_v[%smidx_368] : memref<1511xi32>
        %sscale_368 = memref.load %in_v[%ssidx_368] : memref<1511xi32>
        %dilation_368 = arith.constant 4 : i32
        %padding_368 = arith.constant 16 : i32
        %ki0_368 = arith.constant 1 : i32
        %ki1_368 = arith.constant 2 : i32
        %ki2_368 = arith.constant 7 : i32
        %pad1_368 = arith.constant 1 : i32
        %outidx_368 = arith.constant 368 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_368, %smean_368, %sscale_368, %dilation_368, %padding_368, %ki0_368, %ki1_368, %ki2_368, %pad1_368, %out_v, %outidx_368) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 369: dilation=4 padding=16 kernel=(1,2,8) padding1=0
        %fidx_369 = arith.constant 369 : index
        %bidx_369 = arith.constant 620 : index
        %smidx_369 = arith.constant 1040 : index
        %ssidx_369 = arith.constant 1460 : index
        %bias_369 = memref.load %in_v[%bidx_369] : memref<1511xi32>
        %smean_369 = memref.load %in_v[%smidx_369] : memref<1511xi32>
        %sscale_369 = memref.load %in_v[%ssidx_369] : memref<1511xi32>
        %dilation_369 = arith.constant 4 : i32
        %padding_369 = arith.constant 16 : i32
        %ki0_369 = arith.constant 1 : i32
        %ki1_369 = arith.constant 2 : i32
        %ki2_369 = arith.constant 8 : i32
        %pad1_369 = arith.constant 0 : i32
        %outidx_369 = arith.constant 369 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_369, %smean_369, %sscale_369, %dilation_369, %padding_369, %ki0_369, %ki1_369, %ki2_369, %pad1_369, %out_v, %outidx_369) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 370: dilation=4 padding=16 kernel=(1,3,4) padding1=1
        %fidx_370 = arith.constant 370 : index
        %bidx_370 = arith.constant 621 : index
        %smidx_370 = arith.constant 1041 : index
        %ssidx_370 = arith.constant 1461 : index
        %bias_370 = memref.load %in_v[%bidx_370] : memref<1511xi32>
        %smean_370 = memref.load %in_v[%smidx_370] : memref<1511xi32>
        %sscale_370 = memref.load %in_v[%ssidx_370] : memref<1511xi32>
        %dilation_370 = arith.constant 4 : i32
        %padding_370 = arith.constant 16 : i32
        %ki0_370 = arith.constant 1 : i32
        %ki1_370 = arith.constant 3 : i32
        %ki2_370 = arith.constant 4 : i32
        %pad1_370 = arith.constant 1 : i32
        %outidx_370 = arith.constant 370 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_370, %smean_370, %sscale_370, %dilation_370, %padding_370, %ki0_370, %ki1_370, %ki2_370, %pad1_370, %out_v, %outidx_370) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 371: dilation=4 padding=16 kernel=(1,3,5) padding1=0
        %fidx_371 = arith.constant 371 : index
        %bidx_371 = arith.constant 622 : index
        %smidx_371 = arith.constant 1042 : index
        %ssidx_371 = arith.constant 1462 : index
        %bias_371 = memref.load %in_v[%bidx_371] : memref<1511xi32>
        %smean_371 = memref.load %in_v[%smidx_371] : memref<1511xi32>
        %sscale_371 = memref.load %in_v[%ssidx_371] : memref<1511xi32>
        %dilation_371 = arith.constant 4 : i32
        %padding_371 = arith.constant 16 : i32
        %ki0_371 = arith.constant 1 : i32
        %ki1_371 = arith.constant 3 : i32
        %ki2_371 = arith.constant 5 : i32
        %pad1_371 = arith.constant 0 : i32
        %outidx_371 = arith.constant 371 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_371, %smean_371, %sscale_371, %dilation_371, %padding_371, %ki0_371, %ki1_371, %ki2_371, %pad1_371, %out_v, %outidx_371) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 372: dilation=4 padding=16 kernel=(1,3,6) padding1=1
        %fidx_372 = arith.constant 372 : index
        %bidx_372 = arith.constant 623 : index
        %smidx_372 = arith.constant 1043 : index
        %ssidx_372 = arith.constant 1463 : index
        %bias_372 = memref.load %in_v[%bidx_372] : memref<1511xi32>
        %smean_372 = memref.load %in_v[%smidx_372] : memref<1511xi32>
        %sscale_372 = memref.load %in_v[%ssidx_372] : memref<1511xi32>
        %dilation_372 = arith.constant 4 : i32
        %padding_372 = arith.constant 16 : i32
        %ki0_372 = arith.constant 1 : i32
        %ki1_372 = arith.constant 3 : i32
        %ki2_372 = arith.constant 6 : i32
        %pad1_372 = arith.constant 1 : i32
        %outidx_372 = arith.constant 372 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_372, %smean_372, %sscale_372, %dilation_372, %padding_372, %ki0_372, %ki1_372, %ki2_372, %pad1_372, %out_v, %outidx_372) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 373: dilation=4 padding=16 kernel=(1,3,7) padding1=0
        %fidx_373 = arith.constant 373 : index
        %bidx_373 = arith.constant 624 : index
        %smidx_373 = arith.constant 1044 : index
        %ssidx_373 = arith.constant 1464 : index
        %bias_373 = memref.load %in_v[%bidx_373] : memref<1511xi32>
        %smean_373 = memref.load %in_v[%smidx_373] : memref<1511xi32>
        %sscale_373 = memref.load %in_v[%ssidx_373] : memref<1511xi32>
        %dilation_373 = arith.constant 4 : i32
        %padding_373 = arith.constant 16 : i32
        %ki0_373 = arith.constant 1 : i32
        %ki1_373 = arith.constant 3 : i32
        %ki2_373 = arith.constant 7 : i32
        %pad1_373 = arith.constant 0 : i32
        %outidx_373 = arith.constant 373 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_373, %smean_373, %sscale_373, %dilation_373, %padding_373, %ki0_373, %ki1_373, %ki2_373, %pad1_373, %out_v, %outidx_373) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 374: dilation=4 padding=16 kernel=(1,3,8) padding1=1
        %fidx_374 = arith.constant 374 : index
        %bidx_374 = arith.constant 625 : index
        %smidx_374 = arith.constant 1045 : index
        %ssidx_374 = arith.constant 1465 : index
        %bias_374 = memref.load %in_v[%bidx_374] : memref<1511xi32>
        %smean_374 = memref.load %in_v[%smidx_374] : memref<1511xi32>
        %sscale_374 = memref.load %in_v[%ssidx_374] : memref<1511xi32>
        %dilation_374 = arith.constant 4 : i32
        %padding_374 = arith.constant 16 : i32
        %ki0_374 = arith.constant 1 : i32
        %ki1_374 = arith.constant 3 : i32
        %ki2_374 = arith.constant 8 : i32
        %pad1_374 = arith.constant 1 : i32
        %outidx_374 = arith.constant 374 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_374, %smean_374, %sscale_374, %dilation_374, %padding_374, %ki0_374, %ki1_374, %ki2_374, %pad1_374, %out_v, %outidx_374) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 375: dilation=4 padding=16 kernel=(1,4,5) padding1=0
        %fidx_375 = arith.constant 375 : index
        %bidx_375 = arith.constant 626 : index
        %smidx_375 = arith.constant 1046 : index
        %ssidx_375 = arith.constant 1466 : index
        %bias_375 = memref.load %in_v[%bidx_375] : memref<1511xi32>
        %smean_375 = memref.load %in_v[%smidx_375] : memref<1511xi32>
        %sscale_375 = memref.load %in_v[%ssidx_375] : memref<1511xi32>
        %dilation_375 = arith.constant 4 : i32
        %padding_375 = arith.constant 16 : i32
        %ki0_375 = arith.constant 1 : i32
        %ki1_375 = arith.constant 4 : i32
        %ki2_375 = arith.constant 5 : i32
        %pad1_375 = arith.constant 0 : i32
        %outidx_375 = arith.constant 375 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_375, %smean_375, %sscale_375, %dilation_375, %padding_375, %ki0_375, %ki1_375, %ki2_375, %pad1_375, %out_v, %outidx_375) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 376: dilation=4 padding=16 kernel=(1,4,6) padding1=1
        %fidx_376 = arith.constant 376 : index
        %bidx_376 = arith.constant 627 : index
        %smidx_376 = arith.constant 1047 : index
        %ssidx_376 = arith.constant 1467 : index
        %bias_376 = memref.load %in_v[%bidx_376] : memref<1511xi32>
        %smean_376 = memref.load %in_v[%smidx_376] : memref<1511xi32>
        %sscale_376 = memref.load %in_v[%ssidx_376] : memref<1511xi32>
        %dilation_376 = arith.constant 4 : i32
        %padding_376 = arith.constant 16 : i32
        %ki0_376 = arith.constant 1 : i32
        %ki1_376 = arith.constant 4 : i32
        %ki2_376 = arith.constant 6 : i32
        %pad1_376 = arith.constant 1 : i32
        %outidx_376 = arith.constant 376 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_376, %smean_376, %sscale_376, %dilation_376, %padding_376, %ki0_376, %ki1_376, %ki2_376, %pad1_376, %out_v, %outidx_376) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 377: dilation=4 padding=16 kernel=(1,4,7) padding1=0
        %fidx_377 = arith.constant 377 : index
        %bidx_377 = arith.constant 628 : index
        %smidx_377 = arith.constant 1048 : index
        %ssidx_377 = arith.constant 1468 : index
        %bias_377 = memref.load %in_v[%bidx_377] : memref<1511xi32>
        %smean_377 = memref.load %in_v[%smidx_377] : memref<1511xi32>
        %sscale_377 = memref.load %in_v[%ssidx_377] : memref<1511xi32>
        %dilation_377 = arith.constant 4 : i32
        %padding_377 = arith.constant 16 : i32
        %ki0_377 = arith.constant 1 : i32
        %ki1_377 = arith.constant 4 : i32
        %ki2_377 = arith.constant 7 : i32
        %pad1_377 = arith.constant 0 : i32
        %outidx_377 = arith.constant 377 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_377, %smean_377, %sscale_377, %dilation_377, %padding_377, %ki0_377, %ki1_377, %ki2_377, %pad1_377, %out_v, %outidx_377) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 378: dilation=4 padding=16 kernel=(1,4,8) padding1=1
        %fidx_378 = arith.constant 378 : index
        %bidx_378 = arith.constant 629 : index
        %smidx_378 = arith.constant 1049 : index
        %ssidx_378 = arith.constant 1469 : index
        %bias_378 = memref.load %in_v[%bidx_378] : memref<1511xi32>
        %smean_378 = memref.load %in_v[%smidx_378] : memref<1511xi32>
        %sscale_378 = memref.load %in_v[%ssidx_378] : memref<1511xi32>
        %dilation_378 = arith.constant 4 : i32
        %padding_378 = arith.constant 16 : i32
        %ki0_378 = arith.constant 1 : i32
        %ki1_378 = arith.constant 4 : i32
        %ki2_378 = arith.constant 8 : i32
        %pad1_378 = arith.constant 1 : i32
        %outidx_378 = arith.constant 378 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_378, %smean_378, %sscale_378, %dilation_378, %padding_378, %ki0_378, %ki1_378, %ki2_378, %pad1_378, %out_v, %outidx_378) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 379: dilation=4 padding=16 kernel=(1,5,6) padding1=0
        %fidx_379 = arith.constant 379 : index
        %bidx_379 = arith.constant 630 : index
        %smidx_379 = arith.constant 1050 : index
        %ssidx_379 = arith.constant 1470 : index
        %bias_379 = memref.load %in_v[%bidx_379] : memref<1511xi32>
        %smean_379 = memref.load %in_v[%smidx_379] : memref<1511xi32>
        %sscale_379 = memref.load %in_v[%ssidx_379] : memref<1511xi32>
        %dilation_379 = arith.constant 4 : i32
        %padding_379 = arith.constant 16 : i32
        %ki0_379 = arith.constant 1 : i32
        %ki1_379 = arith.constant 5 : i32
        %ki2_379 = arith.constant 6 : i32
        %pad1_379 = arith.constant 0 : i32
        %outidx_379 = arith.constant 379 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_379, %smean_379, %sscale_379, %dilation_379, %padding_379, %ki0_379, %ki1_379, %ki2_379, %pad1_379, %out_v, %outidx_379) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 380: dilation=4 padding=16 kernel=(1,5,7) padding1=1
        %fidx_380 = arith.constant 380 : index
        %bidx_380 = arith.constant 631 : index
        %smidx_380 = arith.constant 1051 : index
        %ssidx_380 = arith.constant 1471 : index
        %bias_380 = memref.load %in_v[%bidx_380] : memref<1511xi32>
        %smean_380 = memref.load %in_v[%smidx_380] : memref<1511xi32>
        %sscale_380 = memref.load %in_v[%ssidx_380] : memref<1511xi32>
        %dilation_380 = arith.constant 4 : i32
        %padding_380 = arith.constant 16 : i32
        %ki0_380 = arith.constant 1 : i32
        %ki1_380 = arith.constant 5 : i32
        %ki2_380 = arith.constant 7 : i32
        %pad1_380 = arith.constant 1 : i32
        %outidx_380 = arith.constant 380 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_380, %smean_380, %sscale_380, %dilation_380, %padding_380, %ki0_380, %ki1_380, %ki2_380, %pad1_380, %out_v, %outidx_380) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 381: dilation=4 padding=16 kernel=(1,5,8) padding1=0
        %fidx_381 = arith.constant 381 : index
        %bidx_381 = arith.constant 632 : index
        %smidx_381 = arith.constant 1052 : index
        %ssidx_381 = arith.constant 1472 : index
        %bias_381 = memref.load %in_v[%bidx_381] : memref<1511xi32>
        %smean_381 = memref.load %in_v[%smidx_381] : memref<1511xi32>
        %sscale_381 = memref.load %in_v[%ssidx_381] : memref<1511xi32>
        %dilation_381 = arith.constant 4 : i32
        %padding_381 = arith.constant 16 : i32
        %ki0_381 = arith.constant 1 : i32
        %ki1_381 = arith.constant 5 : i32
        %ki2_381 = arith.constant 8 : i32
        %pad1_381 = arith.constant 0 : i32
        %outidx_381 = arith.constant 381 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_381, %smean_381, %sscale_381, %dilation_381, %padding_381, %ki0_381, %ki1_381, %ki2_381, %pad1_381, %out_v, %outidx_381) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 382: dilation=4 padding=16 kernel=(1,6,7) padding1=1
        %fidx_382 = arith.constant 382 : index
        %bidx_382 = arith.constant 633 : index
        %smidx_382 = arith.constant 1053 : index
        %ssidx_382 = arith.constant 1473 : index
        %bias_382 = memref.load %in_v[%bidx_382] : memref<1511xi32>
        %smean_382 = memref.load %in_v[%smidx_382] : memref<1511xi32>
        %sscale_382 = memref.load %in_v[%ssidx_382] : memref<1511xi32>
        %dilation_382 = arith.constant 4 : i32
        %padding_382 = arith.constant 16 : i32
        %ki0_382 = arith.constant 1 : i32
        %ki1_382 = arith.constant 6 : i32
        %ki2_382 = arith.constant 7 : i32
        %pad1_382 = arith.constant 1 : i32
        %outidx_382 = arith.constant 382 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_382, %smean_382, %sscale_382, %dilation_382, %padding_382, %ki0_382, %ki1_382, %ki2_382, %pad1_382, %out_v, %outidx_382) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 383: dilation=4 padding=16 kernel=(1,6,8) padding1=0
        %fidx_383 = arith.constant 383 : index
        %bidx_383 = arith.constant 634 : index
        %smidx_383 = arith.constant 1054 : index
        %ssidx_383 = arith.constant 1474 : index
        %bias_383 = memref.load %in_v[%bidx_383] : memref<1511xi32>
        %smean_383 = memref.load %in_v[%smidx_383] : memref<1511xi32>
        %sscale_383 = memref.load %in_v[%ssidx_383] : memref<1511xi32>
        %dilation_383 = arith.constant 4 : i32
        %padding_383 = arith.constant 16 : i32
        %ki0_383 = arith.constant 1 : i32
        %ki1_383 = arith.constant 6 : i32
        %ki2_383 = arith.constant 8 : i32
        %pad1_383 = arith.constant 0 : i32
        %outidx_383 = arith.constant 383 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_383, %smean_383, %sscale_383, %dilation_383, %padding_383, %ki0_383, %ki1_383, %ki2_383, %pad1_383, %out_v, %outidx_383) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 384: dilation=4 padding=16 kernel=(1,7,8) padding1=1
        %fidx_384 = arith.constant 384 : index
        %bidx_384 = arith.constant 635 : index
        %smidx_384 = arith.constant 1055 : index
        %ssidx_384 = arith.constant 1475 : index
        %bias_384 = memref.load %in_v[%bidx_384] : memref<1511xi32>
        %smean_384 = memref.load %in_v[%smidx_384] : memref<1511xi32>
        %sscale_384 = memref.load %in_v[%ssidx_384] : memref<1511xi32>
        %dilation_384 = arith.constant 4 : i32
        %padding_384 = arith.constant 16 : i32
        %ki0_384 = arith.constant 1 : i32
        %ki1_384 = arith.constant 7 : i32
        %ki2_384 = arith.constant 8 : i32
        %pad1_384 = arith.constant 1 : i32
        %outidx_384 = arith.constant 384 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_384, %smean_384, %sscale_384, %dilation_384, %padding_384, %ki0_384, %ki1_384, %ki2_384, %pad1_384, %out_v, %outidx_384) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 385: dilation=4 padding=16 kernel=(2,3,4) padding1=0
        %fidx_385 = arith.constant 385 : index
        %bidx_385 = arith.constant 636 : index
        %smidx_385 = arith.constant 1056 : index
        %ssidx_385 = arith.constant 1476 : index
        %bias_385 = memref.load %in_v[%bidx_385] : memref<1511xi32>
        %smean_385 = memref.load %in_v[%smidx_385] : memref<1511xi32>
        %sscale_385 = memref.load %in_v[%ssidx_385] : memref<1511xi32>
        %dilation_385 = arith.constant 4 : i32
        %padding_385 = arith.constant 16 : i32
        %ki0_385 = arith.constant 2 : i32
        %ki1_385 = arith.constant 3 : i32
        %ki2_385 = arith.constant 4 : i32
        %pad1_385 = arith.constant 0 : i32
        %outidx_385 = arith.constant 385 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_385, %smean_385, %sscale_385, %dilation_385, %padding_385, %ki0_385, %ki1_385, %ki2_385, %pad1_385, %out_v, %outidx_385) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 386: dilation=4 padding=16 kernel=(2,3,5) padding1=1
        %fidx_386 = arith.constant 386 : index
        %bidx_386 = arith.constant 637 : index
        %smidx_386 = arith.constant 1057 : index
        %ssidx_386 = arith.constant 1477 : index
        %bias_386 = memref.load %in_v[%bidx_386] : memref<1511xi32>
        %smean_386 = memref.load %in_v[%smidx_386] : memref<1511xi32>
        %sscale_386 = memref.load %in_v[%ssidx_386] : memref<1511xi32>
        %dilation_386 = arith.constant 4 : i32
        %padding_386 = arith.constant 16 : i32
        %ki0_386 = arith.constant 2 : i32
        %ki1_386 = arith.constant 3 : i32
        %ki2_386 = arith.constant 5 : i32
        %pad1_386 = arith.constant 1 : i32
        %outidx_386 = arith.constant 386 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_386, %smean_386, %sscale_386, %dilation_386, %padding_386, %ki0_386, %ki1_386, %ki2_386, %pad1_386, %out_v, %outidx_386) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 387: dilation=4 padding=16 kernel=(2,3,6) padding1=0
        %fidx_387 = arith.constant 387 : index
        %bidx_387 = arith.constant 638 : index
        %smidx_387 = arith.constant 1058 : index
        %ssidx_387 = arith.constant 1478 : index
        %bias_387 = memref.load %in_v[%bidx_387] : memref<1511xi32>
        %smean_387 = memref.load %in_v[%smidx_387] : memref<1511xi32>
        %sscale_387 = memref.load %in_v[%ssidx_387] : memref<1511xi32>
        %dilation_387 = arith.constant 4 : i32
        %padding_387 = arith.constant 16 : i32
        %ki0_387 = arith.constant 2 : i32
        %ki1_387 = arith.constant 3 : i32
        %ki2_387 = arith.constant 6 : i32
        %pad1_387 = arith.constant 0 : i32
        %outidx_387 = arith.constant 387 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_387, %smean_387, %sscale_387, %dilation_387, %padding_387, %ki0_387, %ki1_387, %ki2_387, %pad1_387, %out_v, %outidx_387) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 388: dilation=4 padding=16 kernel=(2,3,7) padding1=1
        %fidx_388 = arith.constant 388 : index
        %bidx_388 = arith.constant 639 : index
        %smidx_388 = arith.constant 1059 : index
        %ssidx_388 = arith.constant 1479 : index
        %bias_388 = memref.load %in_v[%bidx_388] : memref<1511xi32>
        %smean_388 = memref.load %in_v[%smidx_388] : memref<1511xi32>
        %sscale_388 = memref.load %in_v[%ssidx_388] : memref<1511xi32>
        %dilation_388 = arith.constant 4 : i32
        %padding_388 = arith.constant 16 : i32
        %ki0_388 = arith.constant 2 : i32
        %ki1_388 = arith.constant 3 : i32
        %ki2_388 = arith.constant 7 : i32
        %pad1_388 = arith.constant 1 : i32
        %outidx_388 = arith.constant 388 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_388, %smean_388, %sscale_388, %dilation_388, %padding_388, %ki0_388, %ki1_388, %ki2_388, %pad1_388, %out_v, %outidx_388) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 389: dilation=4 padding=16 kernel=(2,3,8) padding1=0
        %fidx_389 = arith.constant 389 : index
        %bidx_389 = arith.constant 640 : index
        %smidx_389 = arith.constant 1060 : index
        %ssidx_389 = arith.constant 1480 : index
        %bias_389 = memref.load %in_v[%bidx_389] : memref<1511xi32>
        %smean_389 = memref.load %in_v[%smidx_389] : memref<1511xi32>
        %sscale_389 = memref.load %in_v[%ssidx_389] : memref<1511xi32>
        %dilation_389 = arith.constant 4 : i32
        %padding_389 = arith.constant 16 : i32
        %ki0_389 = arith.constant 2 : i32
        %ki1_389 = arith.constant 3 : i32
        %ki2_389 = arith.constant 8 : i32
        %pad1_389 = arith.constant 0 : i32
        %outidx_389 = arith.constant 389 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_389, %smean_389, %sscale_389, %dilation_389, %padding_389, %ki0_389, %ki1_389, %ki2_389, %pad1_389, %out_v, %outidx_389) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 390: dilation=4 padding=16 kernel=(2,4,5) padding1=1
        %fidx_390 = arith.constant 390 : index
        %bidx_390 = arith.constant 641 : index
        %smidx_390 = arith.constant 1061 : index
        %ssidx_390 = arith.constant 1481 : index
        %bias_390 = memref.load %in_v[%bidx_390] : memref<1511xi32>
        %smean_390 = memref.load %in_v[%smidx_390] : memref<1511xi32>
        %sscale_390 = memref.load %in_v[%ssidx_390] : memref<1511xi32>
        %dilation_390 = arith.constant 4 : i32
        %padding_390 = arith.constant 16 : i32
        %ki0_390 = arith.constant 2 : i32
        %ki1_390 = arith.constant 4 : i32
        %ki2_390 = arith.constant 5 : i32
        %pad1_390 = arith.constant 1 : i32
        %outidx_390 = arith.constant 390 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_390, %smean_390, %sscale_390, %dilation_390, %padding_390, %ki0_390, %ki1_390, %ki2_390, %pad1_390, %out_v, %outidx_390) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 391: dilation=4 padding=16 kernel=(2,4,6) padding1=0
        %fidx_391 = arith.constant 391 : index
        %bidx_391 = arith.constant 642 : index
        %smidx_391 = arith.constant 1062 : index
        %ssidx_391 = arith.constant 1482 : index
        %bias_391 = memref.load %in_v[%bidx_391] : memref<1511xi32>
        %smean_391 = memref.load %in_v[%smidx_391] : memref<1511xi32>
        %sscale_391 = memref.load %in_v[%ssidx_391] : memref<1511xi32>
        %dilation_391 = arith.constant 4 : i32
        %padding_391 = arith.constant 16 : i32
        %ki0_391 = arith.constant 2 : i32
        %ki1_391 = arith.constant 4 : i32
        %ki2_391 = arith.constant 6 : i32
        %pad1_391 = arith.constant 0 : i32
        %outidx_391 = arith.constant 391 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_391, %smean_391, %sscale_391, %dilation_391, %padding_391, %ki0_391, %ki1_391, %ki2_391, %pad1_391, %out_v, %outidx_391) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 392: dilation=4 padding=16 kernel=(2,4,7) padding1=1
        %fidx_392 = arith.constant 392 : index
        %bidx_392 = arith.constant 643 : index
        %smidx_392 = arith.constant 1063 : index
        %ssidx_392 = arith.constant 1483 : index
        %bias_392 = memref.load %in_v[%bidx_392] : memref<1511xi32>
        %smean_392 = memref.load %in_v[%smidx_392] : memref<1511xi32>
        %sscale_392 = memref.load %in_v[%ssidx_392] : memref<1511xi32>
        %dilation_392 = arith.constant 4 : i32
        %padding_392 = arith.constant 16 : i32
        %ki0_392 = arith.constant 2 : i32
        %ki1_392 = arith.constant 4 : i32
        %ki2_392 = arith.constant 7 : i32
        %pad1_392 = arith.constant 1 : i32
        %outidx_392 = arith.constant 392 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_392, %smean_392, %sscale_392, %dilation_392, %padding_392, %ki0_392, %ki1_392, %ki2_392, %pad1_392, %out_v, %outidx_392) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 393: dilation=4 padding=16 kernel=(2,4,8) padding1=0
        %fidx_393 = arith.constant 393 : index
        %bidx_393 = arith.constant 644 : index
        %smidx_393 = arith.constant 1064 : index
        %ssidx_393 = arith.constant 1484 : index
        %bias_393 = memref.load %in_v[%bidx_393] : memref<1511xi32>
        %smean_393 = memref.load %in_v[%smidx_393] : memref<1511xi32>
        %sscale_393 = memref.load %in_v[%ssidx_393] : memref<1511xi32>
        %dilation_393 = arith.constant 4 : i32
        %padding_393 = arith.constant 16 : i32
        %ki0_393 = arith.constant 2 : i32
        %ki1_393 = arith.constant 4 : i32
        %ki2_393 = arith.constant 8 : i32
        %pad1_393 = arith.constant 0 : i32
        %outidx_393 = arith.constant 393 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_393, %smean_393, %sscale_393, %dilation_393, %padding_393, %ki0_393, %ki1_393, %ki2_393, %pad1_393, %out_v, %outidx_393) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 394: dilation=4 padding=16 kernel=(2,5,6) padding1=1
        %fidx_394 = arith.constant 394 : index
        %bidx_394 = arith.constant 645 : index
        %smidx_394 = arith.constant 1065 : index
        %ssidx_394 = arith.constant 1485 : index
        %bias_394 = memref.load %in_v[%bidx_394] : memref<1511xi32>
        %smean_394 = memref.load %in_v[%smidx_394] : memref<1511xi32>
        %sscale_394 = memref.load %in_v[%ssidx_394] : memref<1511xi32>
        %dilation_394 = arith.constant 4 : i32
        %padding_394 = arith.constant 16 : i32
        %ki0_394 = arith.constant 2 : i32
        %ki1_394 = arith.constant 5 : i32
        %ki2_394 = arith.constant 6 : i32
        %pad1_394 = arith.constant 1 : i32
        %outidx_394 = arith.constant 394 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_394, %smean_394, %sscale_394, %dilation_394, %padding_394, %ki0_394, %ki1_394, %ki2_394, %pad1_394, %out_v, %outidx_394) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 395: dilation=4 padding=16 kernel=(2,5,7) padding1=0
        %fidx_395 = arith.constant 395 : index
        %bidx_395 = arith.constant 646 : index
        %smidx_395 = arith.constant 1066 : index
        %ssidx_395 = arith.constant 1486 : index
        %bias_395 = memref.load %in_v[%bidx_395] : memref<1511xi32>
        %smean_395 = memref.load %in_v[%smidx_395] : memref<1511xi32>
        %sscale_395 = memref.load %in_v[%ssidx_395] : memref<1511xi32>
        %dilation_395 = arith.constant 4 : i32
        %padding_395 = arith.constant 16 : i32
        %ki0_395 = arith.constant 2 : i32
        %ki1_395 = arith.constant 5 : i32
        %ki2_395 = arith.constant 7 : i32
        %pad1_395 = arith.constant 0 : i32
        %outidx_395 = arith.constant 395 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_395, %smean_395, %sscale_395, %dilation_395, %padding_395, %ki0_395, %ki1_395, %ki2_395, %pad1_395, %out_v, %outidx_395) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 396: dilation=4 padding=16 kernel=(2,5,8) padding1=1
        %fidx_396 = arith.constant 396 : index
        %bidx_396 = arith.constant 647 : index
        %smidx_396 = arith.constant 1067 : index
        %ssidx_396 = arith.constant 1487 : index
        %bias_396 = memref.load %in_v[%bidx_396] : memref<1511xi32>
        %smean_396 = memref.load %in_v[%smidx_396] : memref<1511xi32>
        %sscale_396 = memref.load %in_v[%ssidx_396] : memref<1511xi32>
        %dilation_396 = arith.constant 4 : i32
        %padding_396 = arith.constant 16 : i32
        %ki0_396 = arith.constant 2 : i32
        %ki1_396 = arith.constant 5 : i32
        %ki2_396 = arith.constant 8 : i32
        %pad1_396 = arith.constant 1 : i32
        %outidx_396 = arith.constant 396 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_396, %smean_396, %sscale_396, %dilation_396, %padding_396, %ki0_396, %ki1_396, %ki2_396, %pad1_396, %out_v, %outidx_396) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 397: dilation=4 padding=16 kernel=(2,6,7) padding1=0
        %fidx_397 = arith.constant 397 : index
        %bidx_397 = arith.constant 648 : index
        %smidx_397 = arith.constant 1068 : index
        %ssidx_397 = arith.constant 1488 : index
        %bias_397 = memref.load %in_v[%bidx_397] : memref<1511xi32>
        %smean_397 = memref.load %in_v[%smidx_397] : memref<1511xi32>
        %sscale_397 = memref.load %in_v[%ssidx_397] : memref<1511xi32>
        %dilation_397 = arith.constant 4 : i32
        %padding_397 = arith.constant 16 : i32
        %ki0_397 = arith.constant 2 : i32
        %ki1_397 = arith.constant 6 : i32
        %ki2_397 = arith.constant 7 : i32
        %pad1_397 = arith.constant 0 : i32
        %outidx_397 = arith.constant 397 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_397, %smean_397, %sscale_397, %dilation_397, %padding_397, %ki0_397, %ki1_397, %ki2_397, %pad1_397, %out_v, %outidx_397) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 398: dilation=4 padding=16 kernel=(2,6,8) padding1=1
        %fidx_398 = arith.constant 398 : index
        %bidx_398 = arith.constant 649 : index
        %smidx_398 = arith.constant 1069 : index
        %ssidx_398 = arith.constant 1489 : index
        %bias_398 = memref.load %in_v[%bidx_398] : memref<1511xi32>
        %smean_398 = memref.load %in_v[%smidx_398] : memref<1511xi32>
        %sscale_398 = memref.load %in_v[%ssidx_398] : memref<1511xi32>
        %dilation_398 = arith.constant 4 : i32
        %padding_398 = arith.constant 16 : i32
        %ki0_398 = arith.constant 2 : i32
        %ki1_398 = arith.constant 6 : i32
        %ki2_398 = arith.constant 8 : i32
        %pad1_398 = arith.constant 1 : i32
        %outidx_398 = arith.constant 398 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_398, %smean_398, %sscale_398, %dilation_398, %padding_398, %ki0_398, %ki1_398, %ki2_398, %pad1_398, %out_v, %outidx_398) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 399: dilation=4 padding=16 kernel=(2,7,8) padding1=0
        %fidx_399 = arith.constant 399 : index
        %bidx_399 = arith.constant 650 : index
        %smidx_399 = arith.constant 1070 : index
        %ssidx_399 = arith.constant 1490 : index
        %bias_399 = memref.load %in_v[%bidx_399] : memref<1511xi32>
        %smean_399 = memref.load %in_v[%smidx_399] : memref<1511xi32>
        %sscale_399 = memref.load %in_v[%ssidx_399] : memref<1511xi32>
        %dilation_399 = arith.constant 4 : i32
        %padding_399 = arith.constant 16 : i32
        %ki0_399 = arith.constant 2 : i32
        %ki1_399 = arith.constant 7 : i32
        %ki2_399 = arith.constant 8 : i32
        %pad1_399 = arith.constant 0 : i32
        %outidx_399 = arith.constant 399 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_399, %smean_399, %sscale_399, %dilation_399, %padding_399, %ki0_399, %ki1_399, %ki2_399, %pad1_399, %out_v, %outidx_399) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 400: dilation=4 padding=16 kernel=(3,4,5) padding1=1
        %fidx_400 = arith.constant 400 : index
        %bidx_400 = arith.constant 651 : index
        %smidx_400 = arith.constant 1071 : index
        %ssidx_400 = arith.constant 1491 : index
        %bias_400 = memref.load %in_v[%bidx_400] : memref<1511xi32>
        %smean_400 = memref.load %in_v[%smidx_400] : memref<1511xi32>
        %sscale_400 = memref.load %in_v[%ssidx_400] : memref<1511xi32>
        %dilation_400 = arith.constant 4 : i32
        %padding_400 = arith.constant 16 : i32
        %ki0_400 = arith.constant 3 : i32
        %ki1_400 = arith.constant 4 : i32
        %ki2_400 = arith.constant 5 : i32
        %pad1_400 = arith.constant 1 : i32
        %outidx_400 = arith.constant 400 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_400, %smean_400, %sscale_400, %dilation_400, %padding_400, %ki0_400, %ki1_400, %ki2_400, %pad1_400, %out_v, %outidx_400) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 401: dilation=4 padding=16 kernel=(3,4,6) padding1=0
        %fidx_401 = arith.constant 401 : index
        %bidx_401 = arith.constant 652 : index
        %smidx_401 = arith.constant 1072 : index
        %ssidx_401 = arith.constant 1492 : index
        %bias_401 = memref.load %in_v[%bidx_401] : memref<1511xi32>
        %smean_401 = memref.load %in_v[%smidx_401] : memref<1511xi32>
        %sscale_401 = memref.load %in_v[%ssidx_401] : memref<1511xi32>
        %dilation_401 = arith.constant 4 : i32
        %padding_401 = arith.constant 16 : i32
        %ki0_401 = arith.constant 3 : i32
        %ki1_401 = arith.constant 4 : i32
        %ki2_401 = arith.constant 6 : i32
        %pad1_401 = arith.constant 0 : i32
        %outidx_401 = arith.constant 401 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_401, %smean_401, %sscale_401, %dilation_401, %padding_401, %ki0_401, %ki1_401, %ki2_401, %pad1_401, %out_v, %outidx_401) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 402: dilation=4 padding=16 kernel=(3,4,7) padding1=1
        %fidx_402 = arith.constant 402 : index
        %bidx_402 = arith.constant 653 : index
        %smidx_402 = arith.constant 1073 : index
        %ssidx_402 = arith.constant 1493 : index
        %bias_402 = memref.load %in_v[%bidx_402] : memref<1511xi32>
        %smean_402 = memref.load %in_v[%smidx_402] : memref<1511xi32>
        %sscale_402 = memref.load %in_v[%ssidx_402] : memref<1511xi32>
        %dilation_402 = arith.constant 4 : i32
        %padding_402 = arith.constant 16 : i32
        %ki0_402 = arith.constant 3 : i32
        %ki1_402 = arith.constant 4 : i32
        %ki2_402 = arith.constant 7 : i32
        %pad1_402 = arith.constant 1 : i32
        %outidx_402 = arith.constant 402 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_402, %smean_402, %sscale_402, %dilation_402, %padding_402, %ki0_402, %ki1_402, %ki2_402, %pad1_402, %out_v, %outidx_402) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 403: dilation=4 padding=16 kernel=(3,4,8) padding1=0
        %fidx_403 = arith.constant 403 : index
        %bidx_403 = arith.constant 654 : index
        %smidx_403 = arith.constant 1074 : index
        %ssidx_403 = arith.constant 1494 : index
        %bias_403 = memref.load %in_v[%bidx_403] : memref<1511xi32>
        %smean_403 = memref.load %in_v[%smidx_403] : memref<1511xi32>
        %sscale_403 = memref.load %in_v[%ssidx_403] : memref<1511xi32>
        %dilation_403 = arith.constant 4 : i32
        %padding_403 = arith.constant 16 : i32
        %ki0_403 = arith.constant 3 : i32
        %ki1_403 = arith.constant 4 : i32
        %ki2_403 = arith.constant 8 : i32
        %pad1_403 = arith.constant 0 : i32
        %outidx_403 = arith.constant 403 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_403, %smean_403, %sscale_403, %dilation_403, %padding_403, %ki0_403, %ki1_403, %ki2_403, %pad1_403, %out_v, %outidx_403) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 404: dilation=4 padding=16 kernel=(3,5,6) padding1=1
        %fidx_404 = arith.constant 404 : index
        %bidx_404 = arith.constant 655 : index
        %smidx_404 = arith.constant 1075 : index
        %ssidx_404 = arith.constant 1495 : index
        %bias_404 = memref.load %in_v[%bidx_404] : memref<1511xi32>
        %smean_404 = memref.load %in_v[%smidx_404] : memref<1511xi32>
        %sscale_404 = memref.load %in_v[%ssidx_404] : memref<1511xi32>
        %dilation_404 = arith.constant 4 : i32
        %padding_404 = arith.constant 16 : i32
        %ki0_404 = arith.constant 3 : i32
        %ki1_404 = arith.constant 5 : i32
        %ki2_404 = arith.constant 6 : i32
        %pad1_404 = arith.constant 1 : i32
        %outidx_404 = arith.constant 404 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_404, %smean_404, %sscale_404, %dilation_404, %padding_404, %ki0_404, %ki1_404, %ki2_404, %pad1_404, %out_v, %outidx_404) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 405: dilation=4 padding=16 kernel=(3,5,7) padding1=0
        %fidx_405 = arith.constant 405 : index
        %bidx_405 = arith.constant 656 : index
        %smidx_405 = arith.constant 1076 : index
        %ssidx_405 = arith.constant 1496 : index
        %bias_405 = memref.load %in_v[%bidx_405] : memref<1511xi32>
        %smean_405 = memref.load %in_v[%smidx_405] : memref<1511xi32>
        %sscale_405 = memref.load %in_v[%ssidx_405] : memref<1511xi32>
        %dilation_405 = arith.constant 4 : i32
        %padding_405 = arith.constant 16 : i32
        %ki0_405 = arith.constant 3 : i32
        %ki1_405 = arith.constant 5 : i32
        %ki2_405 = arith.constant 7 : i32
        %pad1_405 = arith.constant 0 : i32
        %outidx_405 = arith.constant 405 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_405, %smean_405, %sscale_405, %dilation_405, %padding_405, %ki0_405, %ki1_405, %ki2_405, %pad1_405, %out_v, %outidx_405) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 406: dilation=4 padding=16 kernel=(3,5,8) padding1=1
        %fidx_406 = arith.constant 406 : index
        %bidx_406 = arith.constant 657 : index
        %smidx_406 = arith.constant 1077 : index
        %ssidx_406 = arith.constant 1497 : index
        %bias_406 = memref.load %in_v[%bidx_406] : memref<1511xi32>
        %smean_406 = memref.load %in_v[%smidx_406] : memref<1511xi32>
        %sscale_406 = memref.load %in_v[%ssidx_406] : memref<1511xi32>
        %dilation_406 = arith.constant 4 : i32
        %padding_406 = arith.constant 16 : i32
        %ki0_406 = arith.constant 3 : i32
        %ki1_406 = arith.constant 5 : i32
        %ki2_406 = arith.constant 8 : i32
        %pad1_406 = arith.constant 1 : i32
        %outidx_406 = arith.constant 406 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_406, %smean_406, %sscale_406, %dilation_406, %padding_406, %ki0_406, %ki1_406, %ki2_406, %pad1_406, %out_v, %outidx_406) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 407: dilation=4 padding=16 kernel=(3,6,7) padding1=0
        %fidx_407 = arith.constant 407 : index
        %bidx_407 = arith.constant 658 : index
        %smidx_407 = arith.constant 1078 : index
        %ssidx_407 = arith.constant 1498 : index
        %bias_407 = memref.load %in_v[%bidx_407] : memref<1511xi32>
        %smean_407 = memref.load %in_v[%smidx_407] : memref<1511xi32>
        %sscale_407 = memref.load %in_v[%ssidx_407] : memref<1511xi32>
        %dilation_407 = arith.constant 4 : i32
        %padding_407 = arith.constant 16 : i32
        %ki0_407 = arith.constant 3 : i32
        %ki1_407 = arith.constant 6 : i32
        %ki2_407 = arith.constant 7 : i32
        %pad1_407 = arith.constant 0 : i32
        %outidx_407 = arith.constant 407 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_407, %smean_407, %sscale_407, %dilation_407, %padding_407, %ki0_407, %ki1_407, %ki2_407, %pad1_407, %out_v, %outidx_407) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 408: dilation=4 padding=16 kernel=(3,6,8) padding1=1
        %fidx_408 = arith.constant 408 : index
        %bidx_408 = arith.constant 659 : index
        %smidx_408 = arith.constant 1079 : index
        %ssidx_408 = arith.constant 1499 : index
        %bias_408 = memref.load %in_v[%bidx_408] : memref<1511xi32>
        %smean_408 = memref.load %in_v[%smidx_408] : memref<1511xi32>
        %sscale_408 = memref.load %in_v[%ssidx_408] : memref<1511xi32>
        %dilation_408 = arith.constant 4 : i32
        %padding_408 = arith.constant 16 : i32
        %ki0_408 = arith.constant 3 : i32
        %ki1_408 = arith.constant 6 : i32
        %ki2_408 = arith.constant 8 : i32
        %pad1_408 = arith.constant 1 : i32
        %outidx_408 = arith.constant 408 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_408, %smean_408, %sscale_408, %dilation_408, %padding_408, %ki0_408, %ki1_408, %ki2_408, %pad1_408, %out_v, %outidx_408) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 409: dilation=4 padding=16 kernel=(3,7,8) padding1=0
        %fidx_409 = arith.constant 409 : index
        %bidx_409 = arith.constant 660 : index
        %smidx_409 = arith.constant 1080 : index
        %ssidx_409 = arith.constant 1500 : index
        %bias_409 = memref.load %in_v[%bidx_409] : memref<1511xi32>
        %smean_409 = memref.load %in_v[%smidx_409] : memref<1511xi32>
        %sscale_409 = memref.load %in_v[%ssidx_409] : memref<1511xi32>
        %dilation_409 = arith.constant 4 : i32
        %padding_409 = arith.constant 16 : i32
        %ki0_409 = arith.constant 3 : i32
        %ki1_409 = arith.constant 7 : i32
        %ki2_409 = arith.constant 8 : i32
        %pad1_409 = arith.constant 0 : i32
        %outidx_409 = arith.constant 409 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_409, %smean_409, %sscale_409, %dilation_409, %padding_409, %ki0_409, %ki1_409, %ki2_409, %pad1_409, %out_v, %outidx_409) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 410: dilation=4 padding=16 kernel=(4,5,6) padding1=1
        %fidx_410 = arith.constant 410 : index
        %bidx_410 = arith.constant 661 : index
        %smidx_410 = arith.constant 1081 : index
        %ssidx_410 = arith.constant 1501 : index
        %bias_410 = memref.load %in_v[%bidx_410] : memref<1511xi32>
        %smean_410 = memref.load %in_v[%smidx_410] : memref<1511xi32>
        %sscale_410 = memref.load %in_v[%ssidx_410] : memref<1511xi32>
        %dilation_410 = arith.constant 4 : i32
        %padding_410 = arith.constant 16 : i32
        %ki0_410 = arith.constant 4 : i32
        %ki1_410 = arith.constant 5 : i32
        %ki2_410 = arith.constant 6 : i32
        %pad1_410 = arith.constant 1 : i32
        %outidx_410 = arith.constant 410 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_410, %smean_410, %sscale_410, %dilation_410, %padding_410, %ki0_410, %ki1_410, %ki2_410, %pad1_410, %out_v, %outidx_410) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 411: dilation=4 padding=16 kernel=(4,5,7) padding1=0
        %fidx_411 = arith.constant 411 : index
        %bidx_411 = arith.constant 662 : index
        %smidx_411 = arith.constant 1082 : index
        %ssidx_411 = arith.constant 1502 : index
        %bias_411 = memref.load %in_v[%bidx_411] : memref<1511xi32>
        %smean_411 = memref.load %in_v[%smidx_411] : memref<1511xi32>
        %sscale_411 = memref.load %in_v[%ssidx_411] : memref<1511xi32>
        %dilation_411 = arith.constant 4 : i32
        %padding_411 = arith.constant 16 : i32
        %ki0_411 = arith.constant 4 : i32
        %ki1_411 = arith.constant 5 : i32
        %ki2_411 = arith.constant 7 : i32
        %pad1_411 = arith.constant 0 : i32
        %outidx_411 = arith.constant 411 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_411, %smean_411, %sscale_411, %dilation_411, %padding_411, %ki0_411, %ki1_411, %ki2_411, %pad1_411, %out_v, %outidx_411) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 412: dilation=4 padding=16 kernel=(4,5,8) padding1=1
        %fidx_412 = arith.constant 412 : index
        %bidx_412 = arith.constant 663 : index
        %smidx_412 = arith.constant 1083 : index
        %ssidx_412 = arith.constant 1503 : index
        %bias_412 = memref.load %in_v[%bidx_412] : memref<1511xi32>
        %smean_412 = memref.load %in_v[%smidx_412] : memref<1511xi32>
        %sscale_412 = memref.load %in_v[%ssidx_412] : memref<1511xi32>
        %dilation_412 = arith.constant 4 : i32
        %padding_412 = arith.constant 16 : i32
        %ki0_412 = arith.constant 4 : i32
        %ki1_412 = arith.constant 5 : i32
        %ki2_412 = arith.constant 8 : i32
        %pad1_412 = arith.constant 1 : i32
        %outidx_412 = arith.constant 412 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_412, %smean_412, %sscale_412, %dilation_412, %padding_412, %ki0_412, %ki1_412, %ki2_412, %pad1_412, %out_v, %outidx_412) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 413: dilation=4 padding=16 kernel=(4,6,7) padding1=0
        %fidx_413 = arith.constant 413 : index
        %bidx_413 = arith.constant 664 : index
        %smidx_413 = arith.constant 1084 : index
        %ssidx_413 = arith.constant 1504 : index
        %bias_413 = memref.load %in_v[%bidx_413] : memref<1511xi32>
        %smean_413 = memref.load %in_v[%smidx_413] : memref<1511xi32>
        %sscale_413 = memref.load %in_v[%ssidx_413] : memref<1511xi32>
        %dilation_413 = arith.constant 4 : i32
        %padding_413 = arith.constant 16 : i32
        %ki0_413 = arith.constant 4 : i32
        %ki1_413 = arith.constant 6 : i32
        %ki2_413 = arith.constant 7 : i32
        %pad1_413 = arith.constant 0 : i32
        %outidx_413 = arith.constant 413 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_413, %smean_413, %sscale_413, %dilation_413, %padding_413, %ki0_413, %ki1_413, %ki2_413, %pad1_413, %out_v, %outidx_413) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 414: dilation=4 padding=16 kernel=(4,6,8) padding1=1
        %fidx_414 = arith.constant 414 : index
        %bidx_414 = arith.constant 665 : index
        %smidx_414 = arith.constant 1085 : index
        %ssidx_414 = arith.constant 1505 : index
        %bias_414 = memref.load %in_v[%bidx_414] : memref<1511xi32>
        %smean_414 = memref.load %in_v[%smidx_414] : memref<1511xi32>
        %sscale_414 = memref.load %in_v[%ssidx_414] : memref<1511xi32>
        %dilation_414 = arith.constant 4 : i32
        %padding_414 = arith.constant 16 : i32
        %ki0_414 = arith.constant 4 : i32
        %ki1_414 = arith.constant 6 : i32
        %ki2_414 = arith.constant 8 : i32
        %pad1_414 = arith.constant 1 : i32
        %outidx_414 = arith.constant 414 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_414, %smean_414, %sscale_414, %dilation_414, %padding_414, %ki0_414, %ki1_414, %ki2_414, %pad1_414, %out_v, %outidx_414) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 415: dilation=4 padding=16 kernel=(4,7,8) padding1=0
        %fidx_415 = arith.constant 415 : index
        %bidx_415 = arith.constant 666 : index
        %smidx_415 = arith.constant 1086 : index
        %ssidx_415 = arith.constant 1506 : index
        %bias_415 = memref.load %in_v[%bidx_415] : memref<1511xi32>
        %smean_415 = memref.load %in_v[%smidx_415] : memref<1511xi32>
        %sscale_415 = memref.load %in_v[%ssidx_415] : memref<1511xi32>
        %dilation_415 = arith.constant 4 : i32
        %padding_415 = arith.constant 16 : i32
        %ki0_415 = arith.constant 4 : i32
        %ki1_415 = arith.constant 7 : i32
        %ki2_415 = arith.constant 8 : i32
        %pad1_415 = arith.constant 0 : i32
        %outidx_415 = arith.constant 415 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_415, %smean_415, %sscale_415, %dilation_415, %padding_415, %ki0_415, %ki1_415, %ki2_415, %pad1_415, %out_v, %outidx_415) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 416: dilation=4 padding=16 kernel=(5,6,7) padding1=1
        %fidx_416 = arith.constant 416 : index
        %bidx_416 = arith.constant 667 : index
        %smidx_416 = arith.constant 1087 : index
        %ssidx_416 = arith.constant 1507 : index
        %bias_416 = memref.load %in_v[%bidx_416] : memref<1511xi32>
        %smean_416 = memref.load %in_v[%smidx_416] : memref<1511xi32>
        %sscale_416 = memref.load %in_v[%ssidx_416] : memref<1511xi32>
        %dilation_416 = arith.constant 4 : i32
        %padding_416 = arith.constant 16 : i32
        %ki0_416 = arith.constant 5 : i32
        %ki1_416 = arith.constant 6 : i32
        %ki2_416 = arith.constant 7 : i32
        %pad1_416 = arith.constant 1 : i32
        %outidx_416 = arith.constant 416 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_416, %smean_416, %sscale_416, %dilation_416, %padding_416, %ki0_416, %ki1_416, %ki2_416, %pad1_416, %out_v, %outidx_416) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 417: dilation=4 padding=16 kernel=(5,6,8) padding1=0
        %fidx_417 = arith.constant 417 : index
        %bidx_417 = arith.constant 668 : index
        %smidx_417 = arith.constant 1088 : index
        %ssidx_417 = arith.constant 1508 : index
        %bias_417 = memref.load %in_v[%bidx_417] : memref<1511xi32>
        %smean_417 = memref.load %in_v[%smidx_417] : memref<1511xi32>
        %sscale_417 = memref.load %in_v[%ssidx_417] : memref<1511xi32>
        %dilation_417 = arith.constant 4 : i32
        %padding_417 = arith.constant 16 : i32
        %ki0_417 = arith.constant 5 : i32
        %ki1_417 = arith.constant 6 : i32
        %ki2_417 = arith.constant 8 : i32
        %pad1_417 = arith.constant 0 : i32
        %outidx_417 = arith.constant 417 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_417, %smean_417, %sscale_417, %dilation_417, %padding_417, %ki0_417, %ki1_417, %ki2_417, %pad1_417, %out_v, %outidx_417) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 418: dilation=4 padding=16 kernel=(5,7,8) padding1=1
        %fidx_418 = arith.constant 418 : index
        %bidx_418 = arith.constant 669 : index
        %smidx_418 = arith.constant 1089 : index
        %ssidx_418 = arith.constant 1509 : index
        %bias_418 = memref.load %in_v[%bidx_418] : memref<1511xi32>
        %smean_418 = memref.load %in_v[%smidx_418] : memref<1511xi32>
        %sscale_418 = memref.load %in_v[%ssidx_418] : memref<1511xi32>
        %dilation_418 = arith.constant 4 : i32
        %padding_418 = arith.constant 16 : i32
        %ki0_418 = arith.constant 5 : i32
        %ki1_418 = arith.constant 7 : i32
        %ki2_418 = arith.constant 8 : i32
        %pad1_418 = arith.constant 1 : i32
        %outidx_418 = arith.constant 418 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_418, %smean_418, %sscale_418, %dilation_418, %padding_418, %ki0_418, %ki1_418, %ki2_418, %pad1_418, %out_v, %outidx_418) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 419: dilation=4 padding=16 kernel=(6,7,8) padding1=0
        %fidx_419 = arith.constant 419 : index
        %bidx_419 = arith.constant 670 : index
        %smidx_419 = arith.constant 1090 : index
        %ssidx_419 = arith.constant 1510 : index
        %bias_419 = memref.load %in_v[%bidx_419] : memref<1511xi32>
        %smean_419 = memref.load %in_v[%smidx_419] : memref<1511xi32>
        %sscale_419 = memref.load %in_v[%ssidx_419] : memref<1511xi32>
        %dilation_419 = arith.constant 4 : i32
        %padding_419 = arith.constant 16 : i32
        %ki0_419 = arith.constant 6 : i32
        %ki1_419 = arith.constant 7 : i32
        %ki2_419 = arith.constant 8 : i32
        %pad1_419 = arith.constant 0 : i32
        %outidx_419 = arith.constant 419 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_419, %smean_419, %sscale_419, %dilation_419, %padding_419, %ki0_419, %ki1_419, %ki2_419, %pad1_419, %out_v, %outidx_419) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        aie.objectfifo.release @in_mem_0(Consume, 1)
        aie.objectfifo.release @out_mem_0(Produce, 1)
      }
      aie.end
    } {stack_size = 4096 : i32, link_with = "minirocket_transform_kernel.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c_series_off = arith.constant 0 : i32

      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %in_elem = aie.objectfifo.acquire @in_mem_1(Consume, 1) : !aie.objectfifosubview<memref<1511xi32>>
        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<1511xi32>> -> memref<1511xi32>
        %out = aie.objectfifo.acquire @out_mem_1(Produce, 1) : !aie.objectfifosubview<memref<420xi32>>
        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<420xi32>> -> memref<420xi32>

        // feature 0: dilation=6 padding=24 kernel=(0,1,2) padding1=0
        %fidx_0 = arith.constant 0 : index
        %bidx_0 = arith.constant 251 : index
        %smidx_0 = arith.constant 671 : index
        %ssidx_0 = arith.constant 1091 : index
        %bias_0 = memref.load %in_v[%bidx_0] : memref<1511xi32>
        %smean_0 = memref.load %in_v[%smidx_0] : memref<1511xi32>
        %sscale_0 = memref.load %in_v[%ssidx_0] : memref<1511xi32>
        %dilation_0 = arith.constant 6 : i32
        %padding_0 = arith.constant 24 : i32
        %ki0_0 = arith.constant 0 : i32
        %ki1_0 = arith.constant 1 : i32
        %ki2_0 = arith.constant 2 : i32
        %pad1_0 = arith.constant 0 : i32
        %outidx_0 = arith.constant 0 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_0, %smean_0, %sscale_0, %dilation_0, %padding_0, %ki0_0, %ki1_0, %ki2_0, %pad1_0, %out_v, %outidx_0) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 1: dilation=6 padding=24 kernel=(0,1,3) padding1=1
        %fidx_1 = arith.constant 1 : index
        %bidx_1 = arith.constant 252 : index
        %smidx_1 = arith.constant 672 : index
        %ssidx_1 = arith.constant 1092 : index
        %bias_1 = memref.load %in_v[%bidx_1] : memref<1511xi32>
        %smean_1 = memref.load %in_v[%smidx_1] : memref<1511xi32>
        %sscale_1 = memref.load %in_v[%ssidx_1] : memref<1511xi32>
        %dilation_1 = arith.constant 6 : i32
        %padding_1 = arith.constant 24 : i32
        %ki0_1 = arith.constant 0 : i32
        %ki1_1 = arith.constant 1 : i32
        %ki2_1 = arith.constant 3 : i32
        %pad1_1 = arith.constant 1 : i32
        %outidx_1 = arith.constant 1 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_1, %smean_1, %sscale_1, %dilation_1, %padding_1, %ki0_1, %ki1_1, %ki2_1, %pad1_1, %out_v, %outidx_1) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 2: dilation=6 padding=24 kernel=(0,1,4) padding1=0
        %fidx_2 = arith.constant 2 : index
        %bidx_2 = arith.constant 253 : index
        %smidx_2 = arith.constant 673 : index
        %ssidx_2 = arith.constant 1093 : index
        %bias_2 = memref.load %in_v[%bidx_2] : memref<1511xi32>
        %smean_2 = memref.load %in_v[%smidx_2] : memref<1511xi32>
        %sscale_2 = memref.load %in_v[%ssidx_2] : memref<1511xi32>
        %dilation_2 = arith.constant 6 : i32
        %padding_2 = arith.constant 24 : i32
        %ki0_2 = arith.constant 0 : i32
        %ki1_2 = arith.constant 1 : i32
        %ki2_2 = arith.constant 4 : i32
        %pad1_2 = arith.constant 0 : i32
        %outidx_2 = arith.constant 2 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_2, %smean_2, %sscale_2, %dilation_2, %padding_2, %ki0_2, %ki1_2, %ki2_2, %pad1_2, %out_v, %outidx_2) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 3: dilation=6 padding=24 kernel=(0,1,5) padding1=1
        %fidx_3 = arith.constant 3 : index
        %bidx_3 = arith.constant 254 : index
        %smidx_3 = arith.constant 674 : index
        %ssidx_3 = arith.constant 1094 : index
        %bias_3 = memref.load %in_v[%bidx_3] : memref<1511xi32>
        %smean_3 = memref.load %in_v[%smidx_3] : memref<1511xi32>
        %sscale_3 = memref.load %in_v[%ssidx_3] : memref<1511xi32>
        %dilation_3 = arith.constant 6 : i32
        %padding_3 = arith.constant 24 : i32
        %ki0_3 = arith.constant 0 : i32
        %ki1_3 = arith.constant 1 : i32
        %ki2_3 = arith.constant 5 : i32
        %pad1_3 = arith.constant 1 : i32
        %outidx_3 = arith.constant 3 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_3, %smean_3, %sscale_3, %dilation_3, %padding_3, %ki0_3, %ki1_3, %ki2_3, %pad1_3, %out_v, %outidx_3) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 4: dilation=6 padding=24 kernel=(0,1,6) padding1=0
        %fidx_4 = arith.constant 4 : index
        %bidx_4 = arith.constant 255 : index
        %smidx_4 = arith.constant 675 : index
        %ssidx_4 = arith.constant 1095 : index
        %bias_4 = memref.load %in_v[%bidx_4] : memref<1511xi32>
        %smean_4 = memref.load %in_v[%smidx_4] : memref<1511xi32>
        %sscale_4 = memref.load %in_v[%ssidx_4] : memref<1511xi32>
        %dilation_4 = arith.constant 6 : i32
        %padding_4 = arith.constant 24 : i32
        %ki0_4 = arith.constant 0 : i32
        %ki1_4 = arith.constant 1 : i32
        %ki2_4 = arith.constant 6 : i32
        %pad1_4 = arith.constant 0 : i32
        %outidx_4 = arith.constant 4 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_4, %smean_4, %sscale_4, %dilation_4, %padding_4, %ki0_4, %ki1_4, %ki2_4, %pad1_4, %out_v, %outidx_4) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 5: dilation=6 padding=24 kernel=(0,1,7) padding1=1
        %fidx_5 = arith.constant 5 : index
        %bidx_5 = arith.constant 256 : index
        %smidx_5 = arith.constant 676 : index
        %ssidx_5 = arith.constant 1096 : index
        %bias_5 = memref.load %in_v[%bidx_5] : memref<1511xi32>
        %smean_5 = memref.load %in_v[%smidx_5] : memref<1511xi32>
        %sscale_5 = memref.load %in_v[%ssidx_5] : memref<1511xi32>
        %dilation_5 = arith.constant 6 : i32
        %padding_5 = arith.constant 24 : i32
        %ki0_5 = arith.constant 0 : i32
        %ki1_5 = arith.constant 1 : i32
        %ki2_5 = arith.constant 7 : i32
        %pad1_5 = arith.constant 1 : i32
        %outidx_5 = arith.constant 5 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_5, %smean_5, %sscale_5, %dilation_5, %padding_5, %ki0_5, %ki1_5, %ki2_5, %pad1_5, %out_v, %outidx_5) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 6: dilation=6 padding=24 kernel=(0,1,8) padding1=0
        %fidx_6 = arith.constant 6 : index
        %bidx_6 = arith.constant 257 : index
        %smidx_6 = arith.constant 677 : index
        %ssidx_6 = arith.constant 1097 : index
        %bias_6 = memref.load %in_v[%bidx_6] : memref<1511xi32>
        %smean_6 = memref.load %in_v[%smidx_6] : memref<1511xi32>
        %sscale_6 = memref.load %in_v[%ssidx_6] : memref<1511xi32>
        %dilation_6 = arith.constant 6 : i32
        %padding_6 = arith.constant 24 : i32
        %ki0_6 = arith.constant 0 : i32
        %ki1_6 = arith.constant 1 : i32
        %ki2_6 = arith.constant 8 : i32
        %pad1_6 = arith.constant 0 : i32
        %outidx_6 = arith.constant 6 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_6, %smean_6, %sscale_6, %dilation_6, %padding_6, %ki0_6, %ki1_6, %ki2_6, %pad1_6, %out_v, %outidx_6) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 7: dilation=6 padding=24 kernel=(0,2,3) padding1=1
        %fidx_7 = arith.constant 7 : index
        %bidx_7 = arith.constant 258 : index
        %smidx_7 = arith.constant 678 : index
        %ssidx_7 = arith.constant 1098 : index
        %bias_7 = memref.load %in_v[%bidx_7] : memref<1511xi32>
        %smean_7 = memref.load %in_v[%smidx_7] : memref<1511xi32>
        %sscale_7 = memref.load %in_v[%ssidx_7] : memref<1511xi32>
        %dilation_7 = arith.constant 6 : i32
        %padding_7 = arith.constant 24 : i32
        %ki0_7 = arith.constant 0 : i32
        %ki1_7 = arith.constant 2 : i32
        %ki2_7 = arith.constant 3 : i32
        %pad1_7 = arith.constant 1 : i32
        %outidx_7 = arith.constant 7 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_7, %smean_7, %sscale_7, %dilation_7, %padding_7, %ki0_7, %ki1_7, %ki2_7, %pad1_7, %out_v, %outidx_7) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 8: dilation=6 padding=24 kernel=(0,2,4) padding1=0
        %fidx_8 = arith.constant 8 : index
        %bidx_8 = arith.constant 259 : index
        %smidx_8 = arith.constant 679 : index
        %ssidx_8 = arith.constant 1099 : index
        %bias_8 = memref.load %in_v[%bidx_8] : memref<1511xi32>
        %smean_8 = memref.load %in_v[%smidx_8] : memref<1511xi32>
        %sscale_8 = memref.load %in_v[%ssidx_8] : memref<1511xi32>
        %dilation_8 = arith.constant 6 : i32
        %padding_8 = arith.constant 24 : i32
        %ki0_8 = arith.constant 0 : i32
        %ki1_8 = arith.constant 2 : i32
        %ki2_8 = arith.constant 4 : i32
        %pad1_8 = arith.constant 0 : i32
        %outidx_8 = arith.constant 8 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_8, %smean_8, %sscale_8, %dilation_8, %padding_8, %ki0_8, %ki1_8, %ki2_8, %pad1_8, %out_v, %outidx_8) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 9: dilation=6 padding=24 kernel=(0,2,5) padding1=1
        %fidx_9 = arith.constant 9 : index
        %bidx_9 = arith.constant 260 : index
        %smidx_9 = arith.constant 680 : index
        %ssidx_9 = arith.constant 1100 : index
        %bias_9 = memref.load %in_v[%bidx_9] : memref<1511xi32>
        %smean_9 = memref.load %in_v[%smidx_9] : memref<1511xi32>
        %sscale_9 = memref.load %in_v[%ssidx_9] : memref<1511xi32>
        %dilation_9 = arith.constant 6 : i32
        %padding_9 = arith.constant 24 : i32
        %ki0_9 = arith.constant 0 : i32
        %ki1_9 = arith.constant 2 : i32
        %ki2_9 = arith.constant 5 : i32
        %pad1_9 = arith.constant 1 : i32
        %outidx_9 = arith.constant 9 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_9, %smean_9, %sscale_9, %dilation_9, %padding_9, %ki0_9, %ki1_9, %ki2_9, %pad1_9, %out_v, %outidx_9) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 10: dilation=6 padding=24 kernel=(0,2,6) padding1=0
        %fidx_10 = arith.constant 10 : index
        %bidx_10 = arith.constant 261 : index
        %smidx_10 = arith.constant 681 : index
        %ssidx_10 = arith.constant 1101 : index
        %bias_10 = memref.load %in_v[%bidx_10] : memref<1511xi32>
        %smean_10 = memref.load %in_v[%smidx_10] : memref<1511xi32>
        %sscale_10 = memref.load %in_v[%ssidx_10] : memref<1511xi32>
        %dilation_10 = arith.constant 6 : i32
        %padding_10 = arith.constant 24 : i32
        %ki0_10 = arith.constant 0 : i32
        %ki1_10 = arith.constant 2 : i32
        %ki2_10 = arith.constant 6 : i32
        %pad1_10 = arith.constant 0 : i32
        %outidx_10 = arith.constant 10 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_10, %smean_10, %sscale_10, %dilation_10, %padding_10, %ki0_10, %ki1_10, %ki2_10, %pad1_10, %out_v, %outidx_10) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 11: dilation=6 padding=24 kernel=(0,2,7) padding1=1
        %fidx_11 = arith.constant 11 : index
        %bidx_11 = arith.constant 262 : index
        %smidx_11 = arith.constant 682 : index
        %ssidx_11 = arith.constant 1102 : index
        %bias_11 = memref.load %in_v[%bidx_11] : memref<1511xi32>
        %smean_11 = memref.load %in_v[%smidx_11] : memref<1511xi32>
        %sscale_11 = memref.load %in_v[%ssidx_11] : memref<1511xi32>
        %dilation_11 = arith.constant 6 : i32
        %padding_11 = arith.constant 24 : i32
        %ki0_11 = arith.constant 0 : i32
        %ki1_11 = arith.constant 2 : i32
        %ki2_11 = arith.constant 7 : i32
        %pad1_11 = arith.constant 1 : i32
        %outidx_11 = arith.constant 11 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_11, %smean_11, %sscale_11, %dilation_11, %padding_11, %ki0_11, %ki1_11, %ki2_11, %pad1_11, %out_v, %outidx_11) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 12: dilation=6 padding=24 kernel=(0,2,8) padding1=0
        %fidx_12 = arith.constant 12 : index
        %bidx_12 = arith.constant 263 : index
        %smidx_12 = arith.constant 683 : index
        %ssidx_12 = arith.constant 1103 : index
        %bias_12 = memref.load %in_v[%bidx_12] : memref<1511xi32>
        %smean_12 = memref.load %in_v[%smidx_12] : memref<1511xi32>
        %sscale_12 = memref.load %in_v[%ssidx_12] : memref<1511xi32>
        %dilation_12 = arith.constant 6 : i32
        %padding_12 = arith.constant 24 : i32
        %ki0_12 = arith.constant 0 : i32
        %ki1_12 = arith.constant 2 : i32
        %ki2_12 = arith.constant 8 : i32
        %pad1_12 = arith.constant 0 : i32
        %outidx_12 = arith.constant 12 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_12, %smean_12, %sscale_12, %dilation_12, %padding_12, %ki0_12, %ki1_12, %ki2_12, %pad1_12, %out_v, %outidx_12) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 13: dilation=6 padding=24 kernel=(0,3,4) padding1=1
        %fidx_13 = arith.constant 13 : index
        %bidx_13 = arith.constant 264 : index
        %smidx_13 = arith.constant 684 : index
        %ssidx_13 = arith.constant 1104 : index
        %bias_13 = memref.load %in_v[%bidx_13] : memref<1511xi32>
        %smean_13 = memref.load %in_v[%smidx_13] : memref<1511xi32>
        %sscale_13 = memref.load %in_v[%ssidx_13] : memref<1511xi32>
        %dilation_13 = arith.constant 6 : i32
        %padding_13 = arith.constant 24 : i32
        %ki0_13 = arith.constant 0 : i32
        %ki1_13 = arith.constant 3 : i32
        %ki2_13 = arith.constant 4 : i32
        %pad1_13 = arith.constant 1 : i32
        %outidx_13 = arith.constant 13 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_13, %smean_13, %sscale_13, %dilation_13, %padding_13, %ki0_13, %ki1_13, %ki2_13, %pad1_13, %out_v, %outidx_13) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 14: dilation=6 padding=24 kernel=(0,3,5) padding1=0
        %fidx_14 = arith.constant 14 : index
        %bidx_14 = arith.constant 265 : index
        %smidx_14 = arith.constant 685 : index
        %ssidx_14 = arith.constant 1105 : index
        %bias_14 = memref.load %in_v[%bidx_14] : memref<1511xi32>
        %smean_14 = memref.load %in_v[%smidx_14] : memref<1511xi32>
        %sscale_14 = memref.load %in_v[%ssidx_14] : memref<1511xi32>
        %dilation_14 = arith.constant 6 : i32
        %padding_14 = arith.constant 24 : i32
        %ki0_14 = arith.constant 0 : i32
        %ki1_14 = arith.constant 3 : i32
        %ki2_14 = arith.constant 5 : i32
        %pad1_14 = arith.constant 0 : i32
        %outidx_14 = arith.constant 14 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_14, %smean_14, %sscale_14, %dilation_14, %padding_14, %ki0_14, %ki1_14, %ki2_14, %pad1_14, %out_v, %outidx_14) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 15: dilation=6 padding=24 kernel=(0,3,6) padding1=1
        %fidx_15 = arith.constant 15 : index
        %bidx_15 = arith.constant 266 : index
        %smidx_15 = arith.constant 686 : index
        %ssidx_15 = arith.constant 1106 : index
        %bias_15 = memref.load %in_v[%bidx_15] : memref<1511xi32>
        %smean_15 = memref.load %in_v[%smidx_15] : memref<1511xi32>
        %sscale_15 = memref.load %in_v[%ssidx_15] : memref<1511xi32>
        %dilation_15 = arith.constant 6 : i32
        %padding_15 = arith.constant 24 : i32
        %ki0_15 = arith.constant 0 : i32
        %ki1_15 = arith.constant 3 : i32
        %ki2_15 = arith.constant 6 : i32
        %pad1_15 = arith.constant 1 : i32
        %outidx_15 = arith.constant 15 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_15, %smean_15, %sscale_15, %dilation_15, %padding_15, %ki0_15, %ki1_15, %ki2_15, %pad1_15, %out_v, %outidx_15) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 16: dilation=6 padding=24 kernel=(0,3,7) padding1=0
        %fidx_16 = arith.constant 16 : index
        %bidx_16 = arith.constant 267 : index
        %smidx_16 = arith.constant 687 : index
        %ssidx_16 = arith.constant 1107 : index
        %bias_16 = memref.load %in_v[%bidx_16] : memref<1511xi32>
        %smean_16 = memref.load %in_v[%smidx_16] : memref<1511xi32>
        %sscale_16 = memref.load %in_v[%ssidx_16] : memref<1511xi32>
        %dilation_16 = arith.constant 6 : i32
        %padding_16 = arith.constant 24 : i32
        %ki0_16 = arith.constant 0 : i32
        %ki1_16 = arith.constant 3 : i32
        %ki2_16 = arith.constant 7 : i32
        %pad1_16 = arith.constant 0 : i32
        %outidx_16 = arith.constant 16 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_16, %smean_16, %sscale_16, %dilation_16, %padding_16, %ki0_16, %ki1_16, %ki2_16, %pad1_16, %out_v, %outidx_16) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 17: dilation=6 padding=24 kernel=(0,3,8) padding1=1
        %fidx_17 = arith.constant 17 : index
        %bidx_17 = arith.constant 268 : index
        %smidx_17 = arith.constant 688 : index
        %ssidx_17 = arith.constant 1108 : index
        %bias_17 = memref.load %in_v[%bidx_17] : memref<1511xi32>
        %smean_17 = memref.load %in_v[%smidx_17] : memref<1511xi32>
        %sscale_17 = memref.load %in_v[%ssidx_17] : memref<1511xi32>
        %dilation_17 = arith.constant 6 : i32
        %padding_17 = arith.constant 24 : i32
        %ki0_17 = arith.constant 0 : i32
        %ki1_17 = arith.constant 3 : i32
        %ki2_17 = arith.constant 8 : i32
        %pad1_17 = arith.constant 1 : i32
        %outidx_17 = arith.constant 17 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_17, %smean_17, %sscale_17, %dilation_17, %padding_17, %ki0_17, %ki1_17, %ki2_17, %pad1_17, %out_v, %outidx_17) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 18: dilation=6 padding=24 kernel=(0,4,5) padding1=0
        %fidx_18 = arith.constant 18 : index
        %bidx_18 = arith.constant 269 : index
        %smidx_18 = arith.constant 689 : index
        %ssidx_18 = arith.constant 1109 : index
        %bias_18 = memref.load %in_v[%bidx_18] : memref<1511xi32>
        %smean_18 = memref.load %in_v[%smidx_18] : memref<1511xi32>
        %sscale_18 = memref.load %in_v[%ssidx_18] : memref<1511xi32>
        %dilation_18 = arith.constant 6 : i32
        %padding_18 = arith.constant 24 : i32
        %ki0_18 = arith.constant 0 : i32
        %ki1_18 = arith.constant 4 : i32
        %ki2_18 = arith.constant 5 : i32
        %pad1_18 = arith.constant 0 : i32
        %outidx_18 = arith.constant 18 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_18, %smean_18, %sscale_18, %dilation_18, %padding_18, %ki0_18, %ki1_18, %ki2_18, %pad1_18, %out_v, %outidx_18) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 19: dilation=6 padding=24 kernel=(0,4,6) padding1=1
        %fidx_19 = arith.constant 19 : index
        %bidx_19 = arith.constant 270 : index
        %smidx_19 = arith.constant 690 : index
        %ssidx_19 = arith.constant 1110 : index
        %bias_19 = memref.load %in_v[%bidx_19] : memref<1511xi32>
        %smean_19 = memref.load %in_v[%smidx_19] : memref<1511xi32>
        %sscale_19 = memref.load %in_v[%ssidx_19] : memref<1511xi32>
        %dilation_19 = arith.constant 6 : i32
        %padding_19 = arith.constant 24 : i32
        %ki0_19 = arith.constant 0 : i32
        %ki1_19 = arith.constant 4 : i32
        %ki2_19 = arith.constant 6 : i32
        %pad1_19 = arith.constant 1 : i32
        %outidx_19 = arith.constant 19 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_19, %smean_19, %sscale_19, %dilation_19, %padding_19, %ki0_19, %ki1_19, %ki2_19, %pad1_19, %out_v, %outidx_19) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 20: dilation=6 padding=24 kernel=(0,4,7) padding1=0
        %fidx_20 = arith.constant 20 : index
        %bidx_20 = arith.constant 271 : index
        %smidx_20 = arith.constant 691 : index
        %ssidx_20 = arith.constant 1111 : index
        %bias_20 = memref.load %in_v[%bidx_20] : memref<1511xi32>
        %smean_20 = memref.load %in_v[%smidx_20] : memref<1511xi32>
        %sscale_20 = memref.load %in_v[%ssidx_20] : memref<1511xi32>
        %dilation_20 = arith.constant 6 : i32
        %padding_20 = arith.constant 24 : i32
        %ki0_20 = arith.constant 0 : i32
        %ki1_20 = arith.constant 4 : i32
        %ki2_20 = arith.constant 7 : i32
        %pad1_20 = arith.constant 0 : i32
        %outidx_20 = arith.constant 20 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_20, %smean_20, %sscale_20, %dilation_20, %padding_20, %ki0_20, %ki1_20, %ki2_20, %pad1_20, %out_v, %outidx_20) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 21: dilation=6 padding=24 kernel=(0,4,8) padding1=1
        %fidx_21 = arith.constant 21 : index
        %bidx_21 = arith.constant 272 : index
        %smidx_21 = arith.constant 692 : index
        %ssidx_21 = arith.constant 1112 : index
        %bias_21 = memref.load %in_v[%bidx_21] : memref<1511xi32>
        %smean_21 = memref.load %in_v[%smidx_21] : memref<1511xi32>
        %sscale_21 = memref.load %in_v[%ssidx_21] : memref<1511xi32>
        %dilation_21 = arith.constant 6 : i32
        %padding_21 = arith.constant 24 : i32
        %ki0_21 = arith.constant 0 : i32
        %ki1_21 = arith.constant 4 : i32
        %ki2_21 = arith.constant 8 : i32
        %pad1_21 = arith.constant 1 : i32
        %outidx_21 = arith.constant 21 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_21, %smean_21, %sscale_21, %dilation_21, %padding_21, %ki0_21, %ki1_21, %ki2_21, %pad1_21, %out_v, %outidx_21) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 22: dilation=6 padding=24 kernel=(0,5,6) padding1=0
        %fidx_22 = arith.constant 22 : index
        %bidx_22 = arith.constant 273 : index
        %smidx_22 = arith.constant 693 : index
        %ssidx_22 = arith.constant 1113 : index
        %bias_22 = memref.load %in_v[%bidx_22] : memref<1511xi32>
        %smean_22 = memref.load %in_v[%smidx_22] : memref<1511xi32>
        %sscale_22 = memref.load %in_v[%ssidx_22] : memref<1511xi32>
        %dilation_22 = arith.constant 6 : i32
        %padding_22 = arith.constant 24 : i32
        %ki0_22 = arith.constant 0 : i32
        %ki1_22 = arith.constant 5 : i32
        %ki2_22 = arith.constant 6 : i32
        %pad1_22 = arith.constant 0 : i32
        %outidx_22 = arith.constant 22 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_22, %smean_22, %sscale_22, %dilation_22, %padding_22, %ki0_22, %ki1_22, %ki2_22, %pad1_22, %out_v, %outidx_22) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 23: dilation=6 padding=24 kernel=(0,5,7) padding1=1
        %fidx_23 = arith.constant 23 : index
        %bidx_23 = arith.constant 274 : index
        %smidx_23 = arith.constant 694 : index
        %ssidx_23 = arith.constant 1114 : index
        %bias_23 = memref.load %in_v[%bidx_23] : memref<1511xi32>
        %smean_23 = memref.load %in_v[%smidx_23] : memref<1511xi32>
        %sscale_23 = memref.load %in_v[%ssidx_23] : memref<1511xi32>
        %dilation_23 = arith.constant 6 : i32
        %padding_23 = arith.constant 24 : i32
        %ki0_23 = arith.constant 0 : i32
        %ki1_23 = arith.constant 5 : i32
        %ki2_23 = arith.constant 7 : i32
        %pad1_23 = arith.constant 1 : i32
        %outidx_23 = arith.constant 23 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_23, %smean_23, %sscale_23, %dilation_23, %padding_23, %ki0_23, %ki1_23, %ki2_23, %pad1_23, %out_v, %outidx_23) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 24: dilation=6 padding=24 kernel=(0,5,8) padding1=0
        %fidx_24 = arith.constant 24 : index
        %bidx_24 = arith.constant 275 : index
        %smidx_24 = arith.constant 695 : index
        %ssidx_24 = arith.constant 1115 : index
        %bias_24 = memref.load %in_v[%bidx_24] : memref<1511xi32>
        %smean_24 = memref.load %in_v[%smidx_24] : memref<1511xi32>
        %sscale_24 = memref.load %in_v[%ssidx_24] : memref<1511xi32>
        %dilation_24 = arith.constant 6 : i32
        %padding_24 = arith.constant 24 : i32
        %ki0_24 = arith.constant 0 : i32
        %ki1_24 = arith.constant 5 : i32
        %ki2_24 = arith.constant 8 : i32
        %pad1_24 = arith.constant 0 : i32
        %outidx_24 = arith.constant 24 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_24, %smean_24, %sscale_24, %dilation_24, %padding_24, %ki0_24, %ki1_24, %ki2_24, %pad1_24, %out_v, %outidx_24) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 25: dilation=6 padding=24 kernel=(0,6,7) padding1=1
        %fidx_25 = arith.constant 25 : index
        %bidx_25 = arith.constant 276 : index
        %smidx_25 = arith.constant 696 : index
        %ssidx_25 = arith.constant 1116 : index
        %bias_25 = memref.load %in_v[%bidx_25] : memref<1511xi32>
        %smean_25 = memref.load %in_v[%smidx_25] : memref<1511xi32>
        %sscale_25 = memref.load %in_v[%ssidx_25] : memref<1511xi32>
        %dilation_25 = arith.constant 6 : i32
        %padding_25 = arith.constant 24 : i32
        %ki0_25 = arith.constant 0 : i32
        %ki1_25 = arith.constant 6 : i32
        %ki2_25 = arith.constant 7 : i32
        %pad1_25 = arith.constant 1 : i32
        %outidx_25 = arith.constant 25 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_25, %smean_25, %sscale_25, %dilation_25, %padding_25, %ki0_25, %ki1_25, %ki2_25, %pad1_25, %out_v, %outidx_25) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 26: dilation=6 padding=24 kernel=(0,6,8) padding1=0
        %fidx_26 = arith.constant 26 : index
        %bidx_26 = arith.constant 277 : index
        %smidx_26 = arith.constant 697 : index
        %ssidx_26 = arith.constant 1117 : index
        %bias_26 = memref.load %in_v[%bidx_26] : memref<1511xi32>
        %smean_26 = memref.load %in_v[%smidx_26] : memref<1511xi32>
        %sscale_26 = memref.load %in_v[%ssidx_26] : memref<1511xi32>
        %dilation_26 = arith.constant 6 : i32
        %padding_26 = arith.constant 24 : i32
        %ki0_26 = arith.constant 0 : i32
        %ki1_26 = arith.constant 6 : i32
        %ki2_26 = arith.constant 8 : i32
        %pad1_26 = arith.constant 0 : i32
        %outidx_26 = arith.constant 26 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_26, %smean_26, %sscale_26, %dilation_26, %padding_26, %ki0_26, %ki1_26, %ki2_26, %pad1_26, %out_v, %outidx_26) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 27: dilation=6 padding=24 kernel=(0,7,8) padding1=1
        %fidx_27 = arith.constant 27 : index
        %bidx_27 = arith.constant 278 : index
        %smidx_27 = arith.constant 698 : index
        %ssidx_27 = arith.constant 1118 : index
        %bias_27 = memref.load %in_v[%bidx_27] : memref<1511xi32>
        %smean_27 = memref.load %in_v[%smidx_27] : memref<1511xi32>
        %sscale_27 = memref.load %in_v[%ssidx_27] : memref<1511xi32>
        %dilation_27 = arith.constant 6 : i32
        %padding_27 = arith.constant 24 : i32
        %ki0_27 = arith.constant 0 : i32
        %ki1_27 = arith.constant 7 : i32
        %ki2_27 = arith.constant 8 : i32
        %pad1_27 = arith.constant 1 : i32
        %outidx_27 = arith.constant 27 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_27, %smean_27, %sscale_27, %dilation_27, %padding_27, %ki0_27, %ki1_27, %ki2_27, %pad1_27, %out_v, %outidx_27) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 28: dilation=6 padding=24 kernel=(1,2,3) padding1=0
        %fidx_28 = arith.constant 28 : index
        %bidx_28 = arith.constant 279 : index
        %smidx_28 = arith.constant 699 : index
        %ssidx_28 = arith.constant 1119 : index
        %bias_28 = memref.load %in_v[%bidx_28] : memref<1511xi32>
        %smean_28 = memref.load %in_v[%smidx_28] : memref<1511xi32>
        %sscale_28 = memref.load %in_v[%ssidx_28] : memref<1511xi32>
        %dilation_28 = arith.constant 6 : i32
        %padding_28 = arith.constant 24 : i32
        %ki0_28 = arith.constant 1 : i32
        %ki1_28 = arith.constant 2 : i32
        %ki2_28 = arith.constant 3 : i32
        %pad1_28 = arith.constant 0 : i32
        %outidx_28 = arith.constant 28 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_28, %smean_28, %sscale_28, %dilation_28, %padding_28, %ki0_28, %ki1_28, %ki2_28, %pad1_28, %out_v, %outidx_28) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 29: dilation=6 padding=24 kernel=(1,2,4) padding1=1
        %fidx_29 = arith.constant 29 : index
        %bidx_29 = arith.constant 280 : index
        %smidx_29 = arith.constant 700 : index
        %ssidx_29 = arith.constant 1120 : index
        %bias_29 = memref.load %in_v[%bidx_29] : memref<1511xi32>
        %smean_29 = memref.load %in_v[%smidx_29] : memref<1511xi32>
        %sscale_29 = memref.load %in_v[%ssidx_29] : memref<1511xi32>
        %dilation_29 = arith.constant 6 : i32
        %padding_29 = arith.constant 24 : i32
        %ki0_29 = arith.constant 1 : i32
        %ki1_29 = arith.constant 2 : i32
        %ki2_29 = arith.constant 4 : i32
        %pad1_29 = arith.constant 1 : i32
        %outidx_29 = arith.constant 29 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_29, %smean_29, %sscale_29, %dilation_29, %padding_29, %ki0_29, %ki1_29, %ki2_29, %pad1_29, %out_v, %outidx_29) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 30: dilation=6 padding=24 kernel=(1,2,5) padding1=0
        %fidx_30 = arith.constant 30 : index
        %bidx_30 = arith.constant 281 : index
        %smidx_30 = arith.constant 701 : index
        %ssidx_30 = arith.constant 1121 : index
        %bias_30 = memref.load %in_v[%bidx_30] : memref<1511xi32>
        %smean_30 = memref.load %in_v[%smidx_30] : memref<1511xi32>
        %sscale_30 = memref.load %in_v[%ssidx_30] : memref<1511xi32>
        %dilation_30 = arith.constant 6 : i32
        %padding_30 = arith.constant 24 : i32
        %ki0_30 = arith.constant 1 : i32
        %ki1_30 = arith.constant 2 : i32
        %ki2_30 = arith.constant 5 : i32
        %pad1_30 = arith.constant 0 : i32
        %outidx_30 = arith.constant 30 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_30, %smean_30, %sscale_30, %dilation_30, %padding_30, %ki0_30, %ki1_30, %ki2_30, %pad1_30, %out_v, %outidx_30) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 31: dilation=6 padding=24 kernel=(1,2,6) padding1=1
        %fidx_31 = arith.constant 31 : index
        %bidx_31 = arith.constant 282 : index
        %smidx_31 = arith.constant 702 : index
        %ssidx_31 = arith.constant 1122 : index
        %bias_31 = memref.load %in_v[%bidx_31] : memref<1511xi32>
        %smean_31 = memref.load %in_v[%smidx_31] : memref<1511xi32>
        %sscale_31 = memref.load %in_v[%ssidx_31] : memref<1511xi32>
        %dilation_31 = arith.constant 6 : i32
        %padding_31 = arith.constant 24 : i32
        %ki0_31 = arith.constant 1 : i32
        %ki1_31 = arith.constant 2 : i32
        %ki2_31 = arith.constant 6 : i32
        %pad1_31 = arith.constant 1 : i32
        %outidx_31 = arith.constant 31 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_31, %smean_31, %sscale_31, %dilation_31, %padding_31, %ki0_31, %ki1_31, %ki2_31, %pad1_31, %out_v, %outidx_31) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 32: dilation=6 padding=24 kernel=(1,2,7) padding1=0
        %fidx_32 = arith.constant 32 : index
        %bidx_32 = arith.constant 283 : index
        %smidx_32 = arith.constant 703 : index
        %ssidx_32 = arith.constant 1123 : index
        %bias_32 = memref.load %in_v[%bidx_32] : memref<1511xi32>
        %smean_32 = memref.load %in_v[%smidx_32] : memref<1511xi32>
        %sscale_32 = memref.load %in_v[%ssidx_32] : memref<1511xi32>
        %dilation_32 = arith.constant 6 : i32
        %padding_32 = arith.constant 24 : i32
        %ki0_32 = arith.constant 1 : i32
        %ki1_32 = arith.constant 2 : i32
        %ki2_32 = arith.constant 7 : i32
        %pad1_32 = arith.constant 0 : i32
        %outidx_32 = arith.constant 32 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_32, %smean_32, %sscale_32, %dilation_32, %padding_32, %ki0_32, %ki1_32, %ki2_32, %pad1_32, %out_v, %outidx_32) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 33: dilation=6 padding=24 kernel=(1,2,8) padding1=1
        %fidx_33 = arith.constant 33 : index
        %bidx_33 = arith.constant 284 : index
        %smidx_33 = arith.constant 704 : index
        %ssidx_33 = arith.constant 1124 : index
        %bias_33 = memref.load %in_v[%bidx_33] : memref<1511xi32>
        %smean_33 = memref.load %in_v[%smidx_33] : memref<1511xi32>
        %sscale_33 = memref.load %in_v[%ssidx_33] : memref<1511xi32>
        %dilation_33 = arith.constant 6 : i32
        %padding_33 = arith.constant 24 : i32
        %ki0_33 = arith.constant 1 : i32
        %ki1_33 = arith.constant 2 : i32
        %ki2_33 = arith.constant 8 : i32
        %pad1_33 = arith.constant 1 : i32
        %outidx_33 = arith.constant 33 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_33, %smean_33, %sscale_33, %dilation_33, %padding_33, %ki0_33, %ki1_33, %ki2_33, %pad1_33, %out_v, %outidx_33) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 34: dilation=6 padding=24 kernel=(1,3,4) padding1=0
        %fidx_34 = arith.constant 34 : index
        %bidx_34 = arith.constant 285 : index
        %smidx_34 = arith.constant 705 : index
        %ssidx_34 = arith.constant 1125 : index
        %bias_34 = memref.load %in_v[%bidx_34] : memref<1511xi32>
        %smean_34 = memref.load %in_v[%smidx_34] : memref<1511xi32>
        %sscale_34 = memref.load %in_v[%ssidx_34] : memref<1511xi32>
        %dilation_34 = arith.constant 6 : i32
        %padding_34 = arith.constant 24 : i32
        %ki0_34 = arith.constant 1 : i32
        %ki1_34 = arith.constant 3 : i32
        %ki2_34 = arith.constant 4 : i32
        %pad1_34 = arith.constant 0 : i32
        %outidx_34 = arith.constant 34 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_34, %smean_34, %sscale_34, %dilation_34, %padding_34, %ki0_34, %ki1_34, %ki2_34, %pad1_34, %out_v, %outidx_34) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 35: dilation=6 padding=24 kernel=(1,3,5) padding1=1
        %fidx_35 = arith.constant 35 : index
        %bidx_35 = arith.constant 286 : index
        %smidx_35 = arith.constant 706 : index
        %ssidx_35 = arith.constant 1126 : index
        %bias_35 = memref.load %in_v[%bidx_35] : memref<1511xi32>
        %smean_35 = memref.load %in_v[%smidx_35] : memref<1511xi32>
        %sscale_35 = memref.load %in_v[%ssidx_35] : memref<1511xi32>
        %dilation_35 = arith.constant 6 : i32
        %padding_35 = arith.constant 24 : i32
        %ki0_35 = arith.constant 1 : i32
        %ki1_35 = arith.constant 3 : i32
        %ki2_35 = arith.constant 5 : i32
        %pad1_35 = arith.constant 1 : i32
        %outidx_35 = arith.constant 35 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_35, %smean_35, %sscale_35, %dilation_35, %padding_35, %ki0_35, %ki1_35, %ki2_35, %pad1_35, %out_v, %outidx_35) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 36: dilation=6 padding=24 kernel=(1,3,6) padding1=0
        %fidx_36 = arith.constant 36 : index
        %bidx_36 = arith.constant 287 : index
        %smidx_36 = arith.constant 707 : index
        %ssidx_36 = arith.constant 1127 : index
        %bias_36 = memref.load %in_v[%bidx_36] : memref<1511xi32>
        %smean_36 = memref.load %in_v[%smidx_36] : memref<1511xi32>
        %sscale_36 = memref.load %in_v[%ssidx_36] : memref<1511xi32>
        %dilation_36 = arith.constant 6 : i32
        %padding_36 = arith.constant 24 : i32
        %ki0_36 = arith.constant 1 : i32
        %ki1_36 = arith.constant 3 : i32
        %ki2_36 = arith.constant 6 : i32
        %pad1_36 = arith.constant 0 : i32
        %outidx_36 = arith.constant 36 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_36, %smean_36, %sscale_36, %dilation_36, %padding_36, %ki0_36, %ki1_36, %ki2_36, %pad1_36, %out_v, %outidx_36) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 37: dilation=6 padding=24 kernel=(1,3,7) padding1=1
        %fidx_37 = arith.constant 37 : index
        %bidx_37 = arith.constant 288 : index
        %smidx_37 = arith.constant 708 : index
        %ssidx_37 = arith.constant 1128 : index
        %bias_37 = memref.load %in_v[%bidx_37] : memref<1511xi32>
        %smean_37 = memref.load %in_v[%smidx_37] : memref<1511xi32>
        %sscale_37 = memref.load %in_v[%ssidx_37] : memref<1511xi32>
        %dilation_37 = arith.constant 6 : i32
        %padding_37 = arith.constant 24 : i32
        %ki0_37 = arith.constant 1 : i32
        %ki1_37 = arith.constant 3 : i32
        %ki2_37 = arith.constant 7 : i32
        %pad1_37 = arith.constant 1 : i32
        %outidx_37 = arith.constant 37 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_37, %smean_37, %sscale_37, %dilation_37, %padding_37, %ki0_37, %ki1_37, %ki2_37, %pad1_37, %out_v, %outidx_37) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 38: dilation=6 padding=24 kernel=(1,3,8) padding1=0
        %fidx_38 = arith.constant 38 : index
        %bidx_38 = arith.constant 289 : index
        %smidx_38 = arith.constant 709 : index
        %ssidx_38 = arith.constant 1129 : index
        %bias_38 = memref.load %in_v[%bidx_38] : memref<1511xi32>
        %smean_38 = memref.load %in_v[%smidx_38] : memref<1511xi32>
        %sscale_38 = memref.load %in_v[%ssidx_38] : memref<1511xi32>
        %dilation_38 = arith.constant 6 : i32
        %padding_38 = arith.constant 24 : i32
        %ki0_38 = arith.constant 1 : i32
        %ki1_38 = arith.constant 3 : i32
        %ki2_38 = arith.constant 8 : i32
        %pad1_38 = arith.constant 0 : i32
        %outidx_38 = arith.constant 38 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_38, %smean_38, %sscale_38, %dilation_38, %padding_38, %ki0_38, %ki1_38, %ki2_38, %pad1_38, %out_v, %outidx_38) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 39: dilation=6 padding=24 kernel=(1,4,5) padding1=1
        %fidx_39 = arith.constant 39 : index
        %bidx_39 = arith.constant 290 : index
        %smidx_39 = arith.constant 710 : index
        %ssidx_39 = arith.constant 1130 : index
        %bias_39 = memref.load %in_v[%bidx_39] : memref<1511xi32>
        %smean_39 = memref.load %in_v[%smidx_39] : memref<1511xi32>
        %sscale_39 = memref.load %in_v[%ssidx_39] : memref<1511xi32>
        %dilation_39 = arith.constant 6 : i32
        %padding_39 = arith.constant 24 : i32
        %ki0_39 = arith.constant 1 : i32
        %ki1_39 = arith.constant 4 : i32
        %ki2_39 = arith.constant 5 : i32
        %pad1_39 = arith.constant 1 : i32
        %outidx_39 = arith.constant 39 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_39, %smean_39, %sscale_39, %dilation_39, %padding_39, %ki0_39, %ki1_39, %ki2_39, %pad1_39, %out_v, %outidx_39) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 40: dilation=6 padding=24 kernel=(1,4,6) padding1=0
        %fidx_40 = arith.constant 40 : index
        %bidx_40 = arith.constant 291 : index
        %smidx_40 = arith.constant 711 : index
        %ssidx_40 = arith.constant 1131 : index
        %bias_40 = memref.load %in_v[%bidx_40] : memref<1511xi32>
        %smean_40 = memref.load %in_v[%smidx_40] : memref<1511xi32>
        %sscale_40 = memref.load %in_v[%ssidx_40] : memref<1511xi32>
        %dilation_40 = arith.constant 6 : i32
        %padding_40 = arith.constant 24 : i32
        %ki0_40 = arith.constant 1 : i32
        %ki1_40 = arith.constant 4 : i32
        %ki2_40 = arith.constant 6 : i32
        %pad1_40 = arith.constant 0 : i32
        %outidx_40 = arith.constant 40 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_40, %smean_40, %sscale_40, %dilation_40, %padding_40, %ki0_40, %ki1_40, %ki2_40, %pad1_40, %out_v, %outidx_40) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 41: dilation=6 padding=24 kernel=(1,4,7) padding1=1
        %fidx_41 = arith.constant 41 : index
        %bidx_41 = arith.constant 292 : index
        %smidx_41 = arith.constant 712 : index
        %ssidx_41 = arith.constant 1132 : index
        %bias_41 = memref.load %in_v[%bidx_41] : memref<1511xi32>
        %smean_41 = memref.load %in_v[%smidx_41] : memref<1511xi32>
        %sscale_41 = memref.load %in_v[%ssidx_41] : memref<1511xi32>
        %dilation_41 = arith.constant 6 : i32
        %padding_41 = arith.constant 24 : i32
        %ki0_41 = arith.constant 1 : i32
        %ki1_41 = arith.constant 4 : i32
        %ki2_41 = arith.constant 7 : i32
        %pad1_41 = arith.constant 1 : i32
        %outidx_41 = arith.constant 41 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_41, %smean_41, %sscale_41, %dilation_41, %padding_41, %ki0_41, %ki1_41, %ki2_41, %pad1_41, %out_v, %outidx_41) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 42: dilation=6 padding=24 kernel=(1,4,8) padding1=0
        %fidx_42 = arith.constant 42 : index
        %bidx_42 = arith.constant 293 : index
        %smidx_42 = arith.constant 713 : index
        %ssidx_42 = arith.constant 1133 : index
        %bias_42 = memref.load %in_v[%bidx_42] : memref<1511xi32>
        %smean_42 = memref.load %in_v[%smidx_42] : memref<1511xi32>
        %sscale_42 = memref.load %in_v[%ssidx_42] : memref<1511xi32>
        %dilation_42 = arith.constant 6 : i32
        %padding_42 = arith.constant 24 : i32
        %ki0_42 = arith.constant 1 : i32
        %ki1_42 = arith.constant 4 : i32
        %ki2_42 = arith.constant 8 : i32
        %pad1_42 = arith.constant 0 : i32
        %outidx_42 = arith.constant 42 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_42, %smean_42, %sscale_42, %dilation_42, %padding_42, %ki0_42, %ki1_42, %ki2_42, %pad1_42, %out_v, %outidx_42) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 43: dilation=6 padding=24 kernel=(1,5,6) padding1=1
        %fidx_43 = arith.constant 43 : index
        %bidx_43 = arith.constant 294 : index
        %smidx_43 = arith.constant 714 : index
        %ssidx_43 = arith.constant 1134 : index
        %bias_43 = memref.load %in_v[%bidx_43] : memref<1511xi32>
        %smean_43 = memref.load %in_v[%smidx_43] : memref<1511xi32>
        %sscale_43 = memref.load %in_v[%ssidx_43] : memref<1511xi32>
        %dilation_43 = arith.constant 6 : i32
        %padding_43 = arith.constant 24 : i32
        %ki0_43 = arith.constant 1 : i32
        %ki1_43 = arith.constant 5 : i32
        %ki2_43 = arith.constant 6 : i32
        %pad1_43 = arith.constant 1 : i32
        %outidx_43 = arith.constant 43 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_43, %smean_43, %sscale_43, %dilation_43, %padding_43, %ki0_43, %ki1_43, %ki2_43, %pad1_43, %out_v, %outidx_43) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 44: dilation=6 padding=24 kernel=(1,5,7) padding1=0
        %fidx_44 = arith.constant 44 : index
        %bidx_44 = arith.constant 295 : index
        %smidx_44 = arith.constant 715 : index
        %ssidx_44 = arith.constant 1135 : index
        %bias_44 = memref.load %in_v[%bidx_44] : memref<1511xi32>
        %smean_44 = memref.load %in_v[%smidx_44] : memref<1511xi32>
        %sscale_44 = memref.load %in_v[%ssidx_44] : memref<1511xi32>
        %dilation_44 = arith.constant 6 : i32
        %padding_44 = arith.constant 24 : i32
        %ki0_44 = arith.constant 1 : i32
        %ki1_44 = arith.constant 5 : i32
        %ki2_44 = arith.constant 7 : i32
        %pad1_44 = arith.constant 0 : i32
        %outidx_44 = arith.constant 44 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_44, %smean_44, %sscale_44, %dilation_44, %padding_44, %ki0_44, %ki1_44, %ki2_44, %pad1_44, %out_v, %outidx_44) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 45: dilation=6 padding=24 kernel=(1,5,8) padding1=1
        %fidx_45 = arith.constant 45 : index
        %bidx_45 = arith.constant 296 : index
        %smidx_45 = arith.constant 716 : index
        %ssidx_45 = arith.constant 1136 : index
        %bias_45 = memref.load %in_v[%bidx_45] : memref<1511xi32>
        %smean_45 = memref.load %in_v[%smidx_45] : memref<1511xi32>
        %sscale_45 = memref.load %in_v[%ssidx_45] : memref<1511xi32>
        %dilation_45 = arith.constant 6 : i32
        %padding_45 = arith.constant 24 : i32
        %ki0_45 = arith.constant 1 : i32
        %ki1_45 = arith.constant 5 : i32
        %ki2_45 = arith.constant 8 : i32
        %pad1_45 = arith.constant 1 : i32
        %outidx_45 = arith.constant 45 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_45, %smean_45, %sscale_45, %dilation_45, %padding_45, %ki0_45, %ki1_45, %ki2_45, %pad1_45, %out_v, %outidx_45) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 46: dilation=6 padding=24 kernel=(1,6,7) padding1=0
        %fidx_46 = arith.constant 46 : index
        %bidx_46 = arith.constant 297 : index
        %smidx_46 = arith.constant 717 : index
        %ssidx_46 = arith.constant 1137 : index
        %bias_46 = memref.load %in_v[%bidx_46] : memref<1511xi32>
        %smean_46 = memref.load %in_v[%smidx_46] : memref<1511xi32>
        %sscale_46 = memref.load %in_v[%ssidx_46] : memref<1511xi32>
        %dilation_46 = arith.constant 6 : i32
        %padding_46 = arith.constant 24 : i32
        %ki0_46 = arith.constant 1 : i32
        %ki1_46 = arith.constant 6 : i32
        %ki2_46 = arith.constant 7 : i32
        %pad1_46 = arith.constant 0 : i32
        %outidx_46 = arith.constant 46 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_46, %smean_46, %sscale_46, %dilation_46, %padding_46, %ki0_46, %ki1_46, %ki2_46, %pad1_46, %out_v, %outidx_46) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 47: dilation=6 padding=24 kernel=(1,6,8) padding1=1
        %fidx_47 = arith.constant 47 : index
        %bidx_47 = arith.constant 298 : index
        %smidx_47 = arith.constant 718 : index
        %ssidx_47 = arith.constant 1138 : index
        %bias_47 = memref.load %in_v[%bidx_47] : memref<1511xi32>
        %smean_47 = memref.load %in_v[%smidx_47] : memref<1511xi32>
        %sscale_47 = memref.load %in_v[%ssidx_47] : memref<1511xi32>
        %dilation_47 = arith.constant 6 : i32
        %padding_47 = arith.constant 24 : i32
        %ki0_47 = arith.constant 1 : i32
        %ki1_47 = arith.constant 6 : i32
        %ki2_47 = arith.constant 8 : i32
        %pad1_47 = arith.constant 1 : i32
        %outidx_47 = arith.constant 47 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_47, %smean_47, %sscale_47, %dilation_47, %padding_47, %ki0_47, %ki1_47, %ki2_47, %pad1_47, %out_v, %outidx_47) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 48: dilation=6 padding=24 kernel=(1,7,8) padding1=0
        %fidx_48 = arith.constant 48 : index
        %bidx_48 = arith.constant 299 : index
        %smidx_48 = arith.constant 719 : index
        %ssidx_48 = arith.constant 1139 : index
        %bias_48 = memref.load %in_v[%bidx_48] : memref<1511xi32>
        %smean_48 = memref.load %in_v[%smidx_48] : memref<1511xi32>
        %sscale_48 = memref.load %in_v[%ssidx_48] : memref<1511xi32>
        %dilation_48 = arith.constant 6 : i32
        %padding_48 = arith.constant 24 : i32
        %ki0_48 = arith.constant 1 : i32
        %ki1_48 = arith.constant 7 : i32
        %ki2_48 = arith.constant 8 : i32
        %pad1_48 = arith.constant 0 : i32
        %outidx_48 = arith.constant 48 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_48, %smean_48, %sscale_48, %dilation_48, %padding_48, %ki0_48, %ki1_48, %ki2_48, %pad1_48, %out_v, %outidx_48) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 49: dilation=6 padding=24 kernel=(2,3,4) padding1=1
        %fidx_49 = arith.constant 49 : index
        %bidx_49 = arith.constant 300 : index
        %smidx_49 = arith.constant 720 : index
        %ssidx_49 = arith.constant 1140 : index
        %bias_49 = memref.load %in_v[%bidx_49] : memref<1511xi32>
        %smean_49 = memref.load %in_v[%smidx_49] : memref<1511xi32>
        %sscale_49 = memref.load %in_v[%ssidx_49] : memref<1511xi32>
        %dilation_49 = arith.constant 6 : i32
        %padding_49 = arith.constant 24 : i32
        %ki0_49 = arith.constant 2 : i32
        %ki1_49 = arith.constant 3 : i32
        %ki2_49 = arith.constant 4 : i32
        %pad1_49 = arith.constant 1 : i32
        %outidx_49 = arith.constant 49 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_49, %smean_49, %sscale_49, %dilation_49, %padding_49, %ki0_49, %ki1_49, %ki2_49, %pad1_49, %out_v, %outidx_49) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 50: dilation=6 padding=24 kernel=(2,3,5) padding1=0
        %fidx_50 = arith.constant 50 : index
        %bidx_50 = arith.constant 301 : index
        %smidx_50 = arith.constant 721 : index
        %ssidx_50 = arith.constant 1141 : index
        %bias_50 = memref.load %in_v[%bidx_50] : memref<1511xi32>
        %smean_50 = memref.load %in_v[%smidx_50] : memref<1511xi32>
        %sscale_50 = memref.load %in_v[%ssidx_50] : memref<1511xi32>
        %dilation_50 = arith.constant 6 : i32
        %padding_50 = arith.constant 24 : i32
        %ki0_50 = arith.constant 2 : i32
        %ki1_50 = arith.constant 3 : i32
        %ki2_50 = arith.constant 5 : i32
        %pad1_50 = arith.constant 0 : i32
        %outidx_50 = arith.constant 50 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_50, %smean_50, %sscale_50, %dilation_50, %padding_50, %ki0_50, %ki1_50, %ki2_50, %pad1_50, %out_v, %outidx_50) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 51: dilation=6 padding=24 kernel=(2,3,6) padding1=1
        %fidx_51 = arith.constant 51 : index
        %bidx_51 = arith.constant 302 : index
        %smidx_51 = arith.constant 722 : index
        %ssidx_51 = arith.constant 1142 : index
        %bias_51 = memref.load %in_v[%bidx_51] : memref<1511xi32>
        %smean_51 = memref.load %in_v[%smidx_51] : memref<1511xi32>
        %sscale_51 = memref.load %in_v[%ssidx_51] : memref<1511xi32>
        %dilation_51 = arith.constant 6 : i32
        %padding_51 = arith.constant 24 : i32
        %ki0_51 = arith.constant 2 : i32
        %ki1_51 = arith.constant 3 : i32
        %ki2_51 = arith.constant 6 : i32
        %pad1_51 = arith.constant 1 : i32
        %outidx_51 = arith.constant 51 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_51, %smean_51, %sscale_51, %dilation_51, %padding_51, %ki0_51, %ki1_51, %ki2_51, %pad1_51, %out_v, %outidx_51) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 52: dilation=6 padding=24 kernel=(2,3,7) padding1=0
        %fidx_52 = arith.constant 52 : index
        %bidx_52 = arith.constant 303 : index
        %smidx_52 = arith.constant 723 : index
        %ssidx_52 = arith.constant 1143 : index
        %bias_52 = memref.load %in_v[%bidx_52] : memref<1511xi32>
        %smean_52 = memref.load %in_v[%smidx_52] : memref<1511xi32>
        %sscale_52 = memref.load %in_v[%ssidx_52] : memref<1511xi32>
        %dilation_52 = arith.constant 6 : i32
        %padding_52 = arith.constant 24 : i32
        %ki0_52 = arith.constant 2 : i32
        %ki1_52 = arith.constant 3 : i32
        %ki2_52 = arith.constant 7 : i32
        %pad1_52 = arith.constant 0 : i32
        %outidx_52 = arith.constant 52 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_52, %smean_52, %sscale_52, %dilation_52, %padding_52, %ki0_52, %ki1_52, %ki2_52, %pad1_52, %out_v, %outidx_52) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 53: dilation=6 padding=24 kernel=(2,3,8) padding1=1
        %fidx_53 = arith.constant 53 : index
        %bidx_53 = arith.constant 304 : index
        %smidx_53 = arith.constant 724 : index
        %ssidx_53 = arith.constant 1144 : index
        %bias_53 = memref.load %in_v[%bidx_53] : memref<1511xi32>
        %smean_53 = memref.load %in_v[%smidx_53] : memref<1511xi32>
        %sscale_53 = memref.load %in_v[%ssidx_53] : memref<1511xi32>
        %dilation_53 = arith.constant 6 : i32
        %padding_53 = arith.constant 24 : i32
        %ki0_53 = arith.constant 2 : i32
        %ki1_53 = arith.constant 3 : i32
        %ki2_53 = arith.constant 8 : i32
        %pad1_53 = arith.constant 1 : i32
        %outidx_53 = arith.constant 53 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_53, %smean_53, %sscale_53, %dilation_53, %padding_53, %ki0_53, %ki1_53, %ki2_53, %pad1_53, %out_v, %outidx_53) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 54: dilation=6 padding=24 kernel=(2,4,5) padding1=0
        %fidx_54 = arith.constant 54 : index
        %bidx_54 = arith.constant 305 : index
        %smidx_54 = arith.constant 725 : index
        %ssidx_54 = arith.constant 1145 : index
        %bias_54 = memref.load %in_v[%bidx_54] : memref<1511xi32>
        %smean_54 = memref.load %in_v[%smidx_54] : memref<1511xi32>
        %sscale_54 = memref.load %in_v[%ssidx_54] : memref<1511xi32>
        %dilation_54 = arith.constant 6 : i32
        %padding_54 = arith.constant 24 : i32
        %ki0_54 = arith.constant 2 : i32
        %ki1_54 = arith.constant 4 : i32
        %ki2_54 = arith.constant 5 : i32
        %pad1_54 = arith.constant 0 : i32
        %outidx_54 = arith.constant 54 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_54, %smean_54, %sscale_54, %dilation_54, %padding_54, %ki0_54, %ki1_54, %ki2_54, %pad1_54, %out_v, %outidx_54) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 55: dilation=6 padding=24 kernel=(2,4,6) padding1=1
        %fidx_55 = arith.constant 55 : index
        %bidx_55 = arith.constant 306 : index
        %smidx_55 = arith.constant 726 : index
        %ssidx_55 = arith.constant 1146 : index
        %bias_55 = memref.load %in_v[%bidx_55] : memref<1511xi32>
        %smean_55 = memref.load %in_v[%smidx_55] : memref<1511xi32>
        %sscale_55 = memref.load %in_v[%ssidx_55] : memref<1511xi32>
        %dilation_55 = arith.constant 6 : i32
        %padding_55 = arith.constant 24 : i32
        %ki0_55 = arith.constant 2 : i32
        %ki1_55 = arith.constant 4 : i32
        %ki2_55 = arith.constant 6 : i32
        %pad1_55 = arith.constant 1 : i32
        %outidx_55 = arith.constant 55 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_55, %smean_55, %sscale_55, %dilation_55, %padding_55, %ki0_55, %ki1_55, %ki2_55, %pad1_55, %out_v, %outidx_55) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 56: dilation=6 padding=24 kernel=(2,4,7) padding1=0
        %fidx_56 = arith.constant 56 : index
        %bidx_56 = arith.constant 307 : index
        %smidx_56 = arith.constant 727 : index
        %ssidx_56 = arith.constant 1147 : index
        %bias_56 = memref.load %in_v[%bidx_56] : memref<1511xi32>
        %smean_56 = memref.load %in_v[%smidx_56] : memref<1511xi32>
        %sscale_56 = memref.load %in_v[%ssidx_56] : memref<1511xi32>
        %dilation_56 = arith.constant 6 : i32
        %padding_56 = arith.constant 24 : i32
        %ki0_56 = arith.constant 2 : i32
        %ki1_56 = arith.constant 4 : i32
        %ki2_56 = arith.constant 7 : i32
        %pad1_56 = arith.constant 0 : i32
        %outidx_56 = arith.constant 56 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_56, %smean_56, %sscale_56, %dilation_56, %padding_56, %ki0_56, %ki1_56, %ki2_56, %pad1_56, %out_v, %outidx_56) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 57: dilation=6 padding=24 kernel=(2,4,8) padding1=1
        %fidx_57 = arith.constant 57 : index
        %bidx_57 = arith.constant 308 : index
        %smidx_57 = arith.constant 728 : index
        %ssidx_57 = arith.constant 1148 : index
        %bias_57 = memref.load %in_v[%bidx_57] : memref<1511xi32>
        %smean_57 = memref.load %in_v[%smidx_57] : memref<1511xi32>
        %sscale_57 = memref.load %in_v[%ssidx_57] : memref<1511xi32>
        %dilation_57 = arith.constant 6 : i32
        %padding_57 = arith.constant 24 : i32
        %ki0_57 = arith.constant 2 : i32
        %ki1_57 = arith.constant 4 : i32
        %ki2_57 = arith.constant 8 : i32
        %pad1_57 = arith.constant 1 : i32
        %outidx_57 = arith.constant 57 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_57, %smean_57, %sscale_57, %dilation_57, %padding_57, %ki0_57, %ki1_57, %ki2_57, %pad1_57, %out_v, %outidx_57) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 58: dilation=6 padding=24 kernel=(2,5,6) padding1=0
        %fidx_58 = arith.constant 58 : index
        %bidx_58 = arith.constant 309 : index
        %smidx_58 = arith.constant 729 : index
        %ssidx_58 = arith.constant 1149 : index
        %bias_58 = memref.load %in_v[%bidx_58] : memref<1511xi32>
        %smean_58 = memref.load %in_v[%smidx_58] : memref<1511xi32>
        %sscale_58 = memref.load %in_v[%ssidx_58] : memref<1511xi32>
        %dilation_58 = arith.constant 6 : i32
        %padding_58 = arith.constant 24 : i32
        %ki0_58 = arith.constant 2 : i32
        %ki1_58 = arith.constant 5 : i32
        %ki2_58 = arith.constant 6 : i32
        %pad1_58 = arith.constant 0 : i32
        %outidx_58 = arith.constant 58 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_58, %smean_58, %sscale_58, %dilation_58, %padding_58, %ki0_58, %ki1_58, %ki2_58, %pad1_58, %out_v, %outidx_58) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 59: dilation=6 padding=24 kernel=(2,5,7) padding1=1
        %fidx_59 = arith.constant 59 : index
        %bidx_59 = arith.constant 310 : index
        %smidx_59 = arith.constant 730 : index
        %ssidx_59 = arith.constant 1150 : index
        %bias_59 = memref.load %in_v[%bidx_59] : memref<1511xi32>
        %smean_59 = memref.load %in_v[%smidx_59] : memref<1511xi32>
        %sscale_59 = memref.load %in_v[%ssidx_59] : memref<1511xi32>
        %dilation_59 = arith.constant 6 : i32
        %padding_59 = arith.constant 24 : i32
        %ki0_59 = arith.constant 2 : i32
        %ki1_59 = arith.constant 5 : i32
        %ki2_59 = arith.constant 7 : i32
        %pad1_59 = arith.constant 1 : i32
        %outidx_59 = arith.constant 59 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_59, %smean_59, %sscale_59, %dilation_59, %padding_59, %ki0_59, %ki1_59, %ki2_59, %pad1_59, %out_v, %outidx_59) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 60: dilation=6 padding=24 kernel=(2,5,8) padding1=0
        %fidx_60 = arith.constant 60 : index
        %bidx_60 = arith.constant 311 : index
        %smidx_60 = arith.constant 731 : index
        %ssidx_60 = arith.constant 1151 : index
        %bias_60 = memref.load %in_v[%bidx_60] : memref<1511xi32>
        %smean_60 = memref.load %in_v[%smidx_60] : memref<1511xi32>
        %sscale_60 = memref.load %in_v[%ssidx_60] : memref<1511xi32>
        %dilation_60 = arith.constant 6 : i32
        %padding_60 = arith.constant 24 : i32
        %ki0_60 = arith.constant 2 : i32
        %ki1_60 = arith.constant 5 : i32
        %ki2_60 = arith.constant 8 : i32
        %pad1_60 = arith.constant 0 : i32
        %outidx_60 = arith.constant 60 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_60, %smean_60, %sscale_60, %dilation_60, %padding_60, %ki0_60, %ki1_60, %ki2_60, %pad1_60, %out_v, %outidx_60) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 61: dilation=6 padding=24 kernel=(2,6,7) padding1=1
        %fidx_61 = arith.constant 61 : index
        %bidx_61 = arith.constant 312 : index
        %smidx_61 = arith.constant 732 : index
        %ssidx_61 = arith.constant 1152 : index
        %bias_61 = memref.load %in_v[%bidx_61] : memref<1511xi32>
        %smean_61 = memref.load %in_v[%smidx_61] : memref<1511xi32>
        %sscale_61 = memref.load %in_v[%ssidx_61] : memref<1511xi32>
        %dilation_61 = arith.constant 6 : i32
        %padding_61 = arith.constant 24 : i32
        %ki0_61 = arith.constant 2 : i32
        %ki1_61 = arith.constant 6 : i32
        %ki2_61 = arith.constant 7 : i32
        %pad1_61 = arith.constant 1 : i32
        %outidx_61 = arith.constant 61 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_61, %smean_61, %sscale_61, %dilation_61, %padding_61, %ki0_61, %ki1_61, %ki2_61, %pad1_61, %out_v, %outidx_61) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 62: dilation=6 padding=24 kernel=(2,6,8) padding1=0
        %fidx_62 = arith.constant 62 : index
        %bidx_62 = arith.constant 313 : index
        %smidx_62 = arith.constant 733 : index
        %ssidx_62 = arith.constant 1153 : index
        %bias_62 = memref.load %in_v[%bidx_62] : memref<1511xi32>
        %smean_62 = memref.load %in_v[%smidx_62] : memref<1511xi32>
        %sscale_62 = memref.load %in_v[%ssidx_62] : memref<1511xi32>
        %dilation_62 = arith.constant 6 : i32
        %padding_62 = arith.constant 24 : i32
        %ki0_62 = arith.constant 2 : i32
        %ki1_62 = arith.constant 6 : i32
        %ki2_62 = arith.constant 8 : i32
        %pad1_62 = arith.constant 0 : i32
        %outidx_62 = arith.constant 62 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_62, %smean_62, %sscale_62, %dilation_62, %padding_62, %ki0_62, %ki1_62, %ki2_62, %pad1_62, %out_v, %outidx_62) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 63: dilation=6 padding=24 kernel=(2,7,8) padding1=1
        %fidx_63 = arith.constant 63 : index
        %bidx_63 = arith.constant 314 : index
        %smidx_63 = arith.constant 734 : index
        %ssidx_63 = arith.constant 1154 : index
        %bias_63 = memref.load %in_v[%bidx_63] : memref<1511xi32>
        %smean_63 = memref.load %in_v[%smidx_63] : memref<1511xi32>
        %sscale_63 = memref.load %in_v[%ssidx_63] : memref<1511xi32>
        %dilation_63 = arith.constant 6 : i32
        %padding_63 = arith.constant 24 : i32
        %ki0_63 = arith.constant 2 : i32
        %ki1_63 = arith.constant 7 : i32
        %ki2_63 = arith.constant 8 : i32
        %pad1_63 = arith.constant 1 : i32
        %outidx_63 = arith.constant 63 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_63, %smean_63, %sscale_63, %dilation_63, %padding_63, %ki0_63, %ki1_63, %ki2_63, %pad1_63, %out_v, %outidx_63) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 64: dilation=6 padding=24 kernel=(3,4,5) padding1=0
        %fidx_64 = arith.constant 64 : index
        %bidx_64 = arith.constant 315 : index
        %smidx_64 = arith.constant 735 : index
        %ssidx_64 = arith.constant 1155 : index
        %bias_64 = memref.load %in_v[%bidx_64] : memref<1511xi32>
        %smean_64 = memref.load %in_v[%smidx_64] : memref<1511xi32>
        %sscale_64 = memref.load %in_v[%ssidx_64] : memref<1511xi32>
        %dilation_64 = arith.constant 6 : i32
        %padding_64 = arith.constant 24 : i32
        %ki0_64 = arith.constant 3 : i32
        %ki1_64 = arith.constant 4 : i32
        %ki2_64 = arith.constant 5 : i32
        %pad1_64 = arith.constant 0 : i32
        %outidx_64 = arith.constant 64 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_64, %smean_64, %sscale_64, %dilation_64, %padding_64, %ki0_64, %ki1_64, %ki2_64, %pad1_64, %out_v, %outidx_64) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 65: dilation=6 padding=24 kernel=(3,4,6) padding1=1
        %fidx_65 = arith.constant 65 : index
        %bidx_65 = arith.constant 316 : index
        %smidx_65 = arith.constant 736 : index
        %ssidx_65 = arith.constant 1156 : index
        %bias_65 = memref.load %in_v[%bidx_65] : memref<1511xi32>
        %smean_65 = memref.load %in_v[%smidx_65] : memref<1511xi32>
        %sscale_65 = memref.load %in_v[%ssidx_65] : memref<1511xi32>
        %dilation_65 = arith.constant 6 : i32
        %padding_65 = arith.constant 24 : i32
        %ki0_65 = arith.constant 3 : i32
        %ki1_65 = arith.constant 4 : i32
        %ki2_65 = arith.constant 6 : i32
        %pad1_65 = arith.constant 1 : i32
        %outidx_65 = arith.constant 65 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_65, %smean_65, %sscale_65, %dilation_65, %padding_65, %ki0_65, %ki1_65, %ki2_65, %pad1_65, %out_v, %outidx_65) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 66: dilation=6 padding=24 kernel=(3,4,7) padding1=0
        %fidx_66 = arith.constant 66 : index
        %bidx_66 = arith.constant 317 : index
        %smidx_66 = arith.constant 737 : index
        %ssidx_66 = arith.constant 1157 : index
        %bias_66 = memref.load %in_v[%bidx_66] : memref<1511xi32>
        %smean_66 = memref.load %in_v[%smidx_66] : memref<1511xi32>
        %sscale_66 = memref.load %in_v[%ssidx_66] : memref<1511xi32>
        %dilation_66 = arith.constant 6 : i32
        %padding_66 = arith.constant 24 : i32
        %ki0_66 = arith.constant 3 : i32
        %ki1_66 = arith.constant 4 : i32
        %ki2_66 = arith.constant 7 : i32
        %pad1_66 = arith.constant 0 : i32
        %outidx_66 = arith.constant 66 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_66, %smean_66, %sscale_66, %dilation_66, %padding_66, %ki0_66, %ki1_66, %ki2_66, %pad1_66, %out_v, %outidx_66) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 67: dilation=6 padding=24 kernel=(3,4,8) padding1=1
        %fidx_67 = arith.constant 67 : index
        %bidx_67 = arith.constant 318 : index
        %smidx_67 = arith.constant 738 : index
        %ssidx_67 = arith.constant 1158 : index
        %bias_67 = memref.load %in_v[%bidx_67] : memref<1511xi32>
        %smean_67 = memref.load %in_v[%smidx_67] : memref<1511xi32>
        %sscale_67 = memref.load %in_v[%ssidx_67] : memref<1511xi32>
        %dilation_67 = arith.constant 6 : i32
        %padding_67 = arith.constant 24 : i32
        %ki0_67 = arith.constant 3 : i32
        %ki1_67 = arith.constant 4 : i32
        %ki2_67 = arith.constant 8 : i32
        %pad1_67 = arith.constant 1 : i32
        %outidx_67 = arith.constant 67 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_67, %smean_67, %sscale_67, %dilation_67, %padding_67, %ki0_67, %ki1_67, %ki2_67, %pad1_67, %out_v, %outidx_67) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 68: dilation=6 padding=24 kernel=(3,5,6) padding1=0
        %fidx_68 = arith.constant 68 : index
        %bidx_68 = arith.constant 319 : index
        %smidx_68 = arith.constant 739 : index
        %ssidx_68 = arith.constant 1159 : index
        %bias_68 = memref.load %in_v[%bidx_68] : memref<1511xi32>
        %smean_68 = memref.load %in_v[%smidx_68] : memref<1511xi32>
        %sscale_68 = memref.load %in_v[%ssidx_68] : memref<1511xi32>
        %dilation_68 = arith.constant 6 : i32
        %padding_68 = arith.constant 24 : i32
        %ki0_68 = arith.constant 3 : i32
        %ki1_68 = arith.constant 5 : i32
        %ki2_68 = arith.constant 6 : i32
        %pad1_68 = arith.constant 0 : i32
        %outidx_68 = arith.constant 68 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_68, %smean_68, %sscale_68, %dilation_68, %padding_68, %ki0_68, %ki1_68, %ki2_68, %pad1_68, %out_v, %outidx_68) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 69: dilation=6 padding=24 kernel=(3,5,7) padding1=1
        %fidx_69 = arith.constant 69 : index
        %bidx_69 = arith.constant 320 : index
        %smidx_69 = arith.constant 740 : index
        %ssidx_69 = arith.constant 1160 : index
        %bias_69 = memref.load %in_v[%bidx_69] : memref<1511xi32>
        %smean_69 = memref.load %in_v[%smidx_69] : memref<1511xi32>
        %sscale_69 = memref.load %in_v[%ssidx_69] : memref<1511xi32>
        %dilation_69 = arith.constant 6 : i32
        %padding_69 = arith.constant 24 : i32
        %ki0_69 = arith.constant 3 : i32
        %ki1_69 = arith.constant 5 : i32
        %ki2_69 = arith.constant 7 : i32
        %pad1_69 = arith.constant 1 : i32
        %outidx_69 = arith.constant 69 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_69, %smean_69, %sscale_69, %dilation_69, %padding_69, %ki0_69, %ki1_69, %ki2_69, %pad1_69, %out_v, %outidx_69) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 70: dilation=6 padding=24 kernel=(3,5,8) padding1=0
        %fidx_70 = arith.constant 70 : index
        %bidx_70 = arith.constant 321 : index
        %smidx_70 = arith.constant 741 : index
        %ssidx_70 = arith.constant 1161 : index
        %bias_70 = memref.load %in_v[%bidx_70] : memref<1511xi32>
        %smean_70 = memref.load %in_v[%smidx_70] : memref<1511xi32>
        %sscale_70 = memref.load %in_v[%ssidx_70] : memref<1511xi32>
        %dilation_70 = arith.constant 6 : i32
        %padding_70 = arith.constant 24 : i32
        %ki0_70 = arith.constant 3 : i32
        %ki1_70 = arith.constant 5 : i32
        %ki2_70 = arith.constant 8 : i32
        %pad1_70 = arith.constant 0 : i32
        %outidx_70 = arith.constant 70 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_70, %smean_70, %sscale_70, %dilation_70, %padding_70, %ki0_70, %ki1_70, %ki2_70, %pad1_70, %out_v, %outidx_70) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 71: dilation=6 padding=24 kernel=(3,6,7) padding1=1
        %fidx_71 = arith.constant 71 : index
        %bidx_71 = arith.constant 322 : index
        %smidx_71 = arith.constant 742 : index
        %ssidx_71 = arith.constant 1162 : index
        %bias_71 = memref.load %in_v[%bidx_71] : memref<1511xi32>
        %smean_71 = memref.load %in_v[%smidx_71] : memref<1511xi32>
        %sscale_71 = memref.load %in_v[%ssidx_71] : memref<1511xi32>
        %dilation_71 = arith.constant 6 : i32
        %padding_71 = arith.constant 24 : i32
        %ki0_71 = arith.constant 3 : i32
        %ki1_71 = arith.constant 6 : i32
        %ki2_71 = arith.constant 7 : i32
        %pad1_71 = arith.constant 1 : i32
        %outidx_71 = arith.constant 71 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_71, %smean_71, %sscale_71, %dilation_71, %padding_71, %ki0_71, %ki1_71, %ki2_71, %pad1_71, %out_v, %outidx_71) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 72: dilation=6 padding=24 kernel=(3,6,8) padding1=0
        %fidx_72 = arith.constant 72 : index
        %bidx_72 = arith.constant 323 : index
        %smidx_72 = arith.constant 743 : index
        %ssidx_72 = arith.constant 1163 : index
        %bias_72 = memref.load %in_v[%bidx_72] : memref<1511xi32>
        %smean_72 = memref.load %in_v[%smidx_72] : memref<1511xi32>
        %sscale_72 = memref.load %in_v[%ssidx_72] : memref<1511xi32>
        %dilation_72 = arith.constant 6 : i32
        %padding_72 = arith.constant 24 : i32
        %ki0_72 = arith.constant 3 : i32
        %ki1_72 = arith.constant 6 : i32
        %ki2_72 = arith.constant 8 : i32
        %pad1_72 = arith.constant 0 : i32
        %outidx_72 = arith.constant 72 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_72, %smean_72, %sscale_72, %dilation_72, %padding_72, %ki0_72, %ki1_72, %ki2_72, %pad1_72, %out_v, %outidx_72) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 73: dilation=6 padding=24 kernel=(3,7,8) padding1=1
        %fidx_73 = arith.constant 73 : index
        %bidx_73 = arith.constant 324 : index
        %smidx_73 = arith.constant 744 : index
        %ssidx_73 = arith.constant 1164 : index
        %bias_73 = memref.load %in_v[%bidx_73] : memref<1511xi32>
        %smean_73 = memref.load %in_v[%smidx_73] : memref<1511xi32>
        %sscale_73 = memref.load %in_v[%ssidx_73] : memref<1511xi32>
        %dilation_73 = arith.constant 6 : i32
        %padding_73 = arith.constant 24 : i32
        %ki0_73 = arith.constant 3 : i32
        %ki1_73 = arith.constant 7 : i32
        %ki2_73 = arith.constant 8 : i32
        %pad1_73 = arith.constant 1 : i32
        %outidx_73 = arith.constant 73 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_73, %smean_73, %sscale_73, %dilation_73, %padding_73, %ki0_73, %ki1_73, %ki2_73, %pad1_73, %out_v, %outidx_73) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 74: dilation=6 padding=24 kernel=(4,5,6) padding1=0
        %fidx_74 = arith.constant 74 : index
        %bidx_74 = arith.constant 325 : index
        %smidx_74 = arith.constant 745 : index
        %ssidx_74 = arith.constant 1165 : index
        %bias_74 = memref.load %in_v[%bidx_74] : memref<1511xi32>
        %smean_74 = memref.load %in_v[%smidx_74] : memref<1511xi32>
        %sscale_74 = memref.load %in_v[%ssidx_74] : memref<1511xi32>
        %dilation_74 = arith.constant 6 : i32
        %padding_74 = arith.constant 24 : i32
        %ki0_74 = arith.constant 4 : i32
        %ki1_74 = arith.constant 5 : i32
        %ki2_74 = arith.constant 6 : i32
        %pad1_74 = arith.constant 0 : i32
        %outidx_74 = arith.constant 74 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_74, %smean_74, %sscale_74, %dilation_74, %padding_74, %ki0_74, %ki1_74, %ki2_74, %pad1_74, %out_v, %outidx_74) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 75: dilation=6 padding=24 kernel=(4,5,7) padding1=1
        %fidx_75 = arith.constant 75 : index
        %bidx_75 = arith.constant 326 : index
        %smidx_75 = arith.constant 746 : index
        %ssidx_75 = arith.constant 1166 : index
        %bias_75 = memref.load %in_v[%bidx_75] : memref<1511xi32>
        %smean_75 = memref.load %in_v[%smidx_75] : memref<1511xi32>
        %sscale_75 = memref.load %in_v[%ssidx_75] : memref<1511xi32>
        %dilation_75 = arith.constant 6 : i32
        %padding_75 = arith.constant 24 : i32
        %ki0_75 = arith.constant 4 : i32
        %ki1_75 = arith.constant 5 : i32
        %ki2_75 = arith.constant 7 : i32
        %pad1_75 = arith.constant 1 : i32
        %outidx_75 = arith.constant 75 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_75, %smean_75, %sscale_75, %dilation_75, %padding_75, %ki0_75, %ki1_75, %ki2_75, %pad1_75, %out_v, %outidx_75) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 76: dilation=6 padding=24 kernel=(4,5,8) padding1=0
        %fidx_76 = arith.constant 76 : index
        %bidx_76 = arith.constant 327 : index
        %smidx_76 = arith.constant 747 : index
        %ssidx_76 = arith.constant 1167 : index
        %bias_76 = memref.load %in_v[%bidx_76] : memref<1511xi32>
        %smean_76 = memref.load %in_v[%smidx_76] : memref<1511xi32>
        %sscale_76 = memref.load %in_v[%ssidx_76] : memref<1511xi32>
        %dilation_76 = arith.constant 6 : i32
        %padding_76 = arith.constant 24 : i32
        %ki0_76 = arith.constant 4 : i32
        %ki1_76 = arith.constant 5 : i32
        %ki2_76 = arith.constant 8 : i32
        %pad1_76 = arith.constant 0 : i32
        %outidx_76 = arith.constant 76 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_76, %smean_76, %sscale_76, %dilation_76, %padding_76, %ki0_76, %ki1_76, %ki2_76, %pad1_76, %out_v, %outidx_76) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 77: dilation=6 padding=24 kernel=(4,6,7) padding1=1
        %fidx_77 = arith.constant 77 : index
        %bidx_77 = arith.constant 328 : index
        %smidx_77 = arith.constant 748 : index
        %ssidx_77 = arith.constant 1168 : index
        %bias_77 = memref.load %in_v[%bidx_77] : memref<1511xi32>
        %smean_77 = memref.load %in_v[%smidx_77] : memref<1511xi32>
        %sscale_77 = memref.load %in_v[%ssidx_77] : memref<1511xi32>
        %dilation_77 = arith.constant 6 : i32
        %padding_77 = arith.constant 24 : i32
        %ki0_77 = arith.constant 4 : i32
        %ki1_77 = arith.constant 6 : i32
        %ki2_77 = arith.constant 7 : i32
        %pad1_77 = arith.constant 1 : i32
        %outidx_77 = arith.constant 77 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_77, %smean_77, %sscale_77, %dilation_77, %padding_77, %ki0_77, %ki1_77, %ki2_77, %pad1_77, %out_v, %outidx_77) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 78: dilation=6 padding=24 kernel=(4,6,8) padding1=0
        %fidx_78 = arith.constant 78 : index
        %bidx_78 = arith.constant 329 : index
        %smidx_78 = arith.constant 749 : index
        %ssidx_78 = arith.constant 1169 : index
        %bias_78 = memref.load %in_v[%bidx_78] : memref<1511xi32>
        %smean_78 = memref.load %in_v[%smidx_78] : memref<1511xi32>
        %sscale_78 = memref.load %in_v[%ssidx_78] : memref<1511xi32>
        %dilation_78 = arith.constant 6 : i32
        %padding_78 = arith.constant 24 : i32
        %ki0_78 = arith.constant 4 : i32
        %ki1_78 = arith.constant 6 : i32
        %ki2_78 = arith.constant 8 : i32
        %pad1_78 = arith.constant 0 : i32
        %outidx_78 = arith.constant 78 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_78, %smean_78, %sscale_78, %dilation_78, %padding_78, %ki0_78, %ki1_78, %ki2_78, %pad1_78, %out_v, %outidx_78) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 79: dilation=6 padding=24 kernel=(4,7,8) padding1=1
        %fidx_79 = arith.constant 79 : index
        %bidx_79 = arith.constant 330 : index
        %smidx_79 = arith.constant 750 : index
        %ssidx_79 = arith.constant 1170 : index
        %bias_79 = memref.load %in_v[%bidx_79] : memref<1511xi32>
        %smean_79 = memref.load %in_v[%smidx_79] : memref<1511xi32>
        %sscale_79 = memref.load %in_v[%ssidx_79] : memref<1511xi32>
        %dilation_79 = arith.constant 6 : i32
        %padding_79 = arith.constant 24 : i32
        %ki0_79 = arith.constant 4 : i32
        %ki1_79 = arith.constant 7 : i32
        %ki2_79 = arith.constant 8 : i32
        %pad1_79 = arith.constant 1 : i32
        %outidx_79 = arith.constant 79 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_79, %smean_79, %sscale_79, %dilation_79, %padding_79, %ki0_79, %ki1_79, %ki2_79, %pad1_79, %out_v, %outidx_79) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 80: dilation=6 padding=24 kernel=(5,6,7) padding1=0
        %fidx_80 = arith.constant 80 : index
        %bidx_80 = arith.constant 331 : index
        %smidx_80 = arith.constant 751 : index
        %ssidx_80 = arith.constant 1171 : index
        %bias_80 = memref.load %in_v[%bidx_80] : memref<1511xi32>
        %smean_80 = memref.load %in_v[%smidx_80] : memref<1511xi32>
        %sscale_80 = memref.load %in_v[%ssidx_80] : memref<1511xi32>
        %dilation_80 = arith.constant 6 : i32
        %padding_80 = arith.constant 24 : i32
        %ki0_80 = arith.constant 5 : i32
        %ki1_80 = arith.constant 6 : i32
        %ki2_80 = arith.constant 7 : i32
        %pad1_80 = arith.constant 0 : i32
        %outidx_80 = arith.constant 80 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_80, %smean_80, %sscale_80, %dilation_80, %padding_80, %ki0_80, %ki1_80, %ki2_80, %pad1_80, %out_v, %outidx_80) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 81: dilation=6 padding=24 kernel=(5,6,8) padding1=1
        %fidx_81 = arith.constant 81 : index
        %bidx_81 = arith.constant 332 : index
        %smidx_81 = arith.constant 752 : index
        %ssidx_81 = arith.constant 1172 : index
        %bias_81 = memref.load %in_v[%bidx_81] : memref<1511xi32>
        %smean_81 = memref.load %in_v[%smidx_81] : memref<1511xi32>
        %sscale_81 = memref.load %in_v[%ssidx_81] : memref<1511xi32>
        %dilation_81 = arith.constant 6 : i32
        %padding_81 = arith.constant 24 : i32
        %ki0_81 = arith.constant 5 : i32
        %ki1_81 = arith.constant 6 : i32
        %ki2_81 = arith.constant 8 : i32
        %pad1_81 = arith.constant 1 : i32
        %outidx_81 = arith.constant 81 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_81, %smean_81, %sscale_81, %dilation_81, %padding_81, %ki0_81, %ki1_81, %ki2_81, %pad1_81, %out_v, %outidx_81) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 82: dilation=6 padding=24 kernel=(5,7,8) padding1=0
        %fidx_82 = arith.constant 82 : index
        %bidx_82 = arith.constant 333 : index
        %smidx_82 = arith.constant 753 : index
        %ssidx_82 = arith.constant 1173 : index
        %bias_82 = memref.load %in_v[%bidx_82] : memref<1511xi32>
        %smean_82 = memref.load %in_v[%smidx_82] : memref<1511xi32>
        %sscale_82 = memref.load %in_v[%ssidx_82] : memref<1511xi32>
        %dilation_82 = arith.constant 6 : i32
        %padding_82 = arith.constant 24 : i32
        %ki0_82 = arith.constant 5 : i32
        %ki1_82 = arith.constant 7 : i32
        %ki2_82 = arith.constant 8 : i32
        %pad1_82 = arith.constant 0 : i32
        %outidx_82 = arith.constant 82 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_82, %smean_82, %sscale_82, %dilation_82, %padding_82, %ki0_82, %ki1_82, %ki2_82, %pad1_82, %out_v, %outidx_82) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 83: dilation=6 padding=24 kernel=(6,7,8) padding1=1
        %fidx_83 = arith.constant 83 : index
        %bidx_83 = arith.constant 334 : index
        %smidx_83 = arith.constant 754 : index
        %ssidx_83 = arith.constant 1174 : index
        %bias_83 = memref.load %in_v[%bidx_83] : memref<1511xi32>
        %smean_83 = memref.load %in_v[%smidx_83] : memref<1511xi32>
        %sscale_83 = memref.load %in_v[%ssidx_83] : memref<1511xi32>
        %dilation_83 = arith.constant 6 : i32
        %padding_83 = arith.constant 24 : i32
        %ki0_83 = arith.constant 6 : i32
        %ki1_83 = arith.constant 7 : i32
        %ki2_83 = arith.constant 8 : i32
        %pad1_83 = arith.constant 1 : i32
        %outidx_83 = arith.constant 83 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_83, %smean_83, %sscale_83, %dilation_83, %padding_83, %ki0_83, %ki1_83, %ki2_83, %pad1_83, %out_v, %outidx_83) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 84: dilation=9 padding=36 kernel=(0,1,2) padding1=1
        %fidx_84 = arith.constant 84 : index
        %bidx_84 = arith.constant 335 : index
        %smidx_84 = arith.constant 755 : index
        %ssidx_84 = arith.constant 1175 : index
        %bias_84 = memref.load %in_v[%bidx_84] : memref<1511xi32>
        %smean_84 = memref.load %in_v[%smidx_84] : memref<1511xi32>
        %sscale_84 = memref.load %in_v[%ssidx_84] : memref<1511xi32>
        %dilation_84 = arith.constant 9 : i32
        %padding_84 = arith.constant 36 : i32
        %ki0_84 = arith.constant 0 : i32
        %ki1_84 = arith.constant 1 : i32
        %ki2_84 = arith.constant 2 : i32
        %pad1_84 = arith.constant 1 : i32
        %outidx_84 = arith.constant 84 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_84, %smean_84, %sscale_84, %dilation_84, %padding_84, %ki0_84, %ki1_84, %ki2_84, %pad1_84, %out_v, %outidx_84) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 85: dilation=9 padding=36 kernel=(0,1,3) padding1=0
        %fidx_85 = arith.constant 85 : index
        %bidx_85 = arith.constant 336 : index
        %smidx_85 = arith.constant 756 : index
        %ssidx_85 = arith.constant 1176 : index
        %bias_85 = memref.load %in_v[%bidx_85] : memref<1511xi32>
        %smean_85 = memref.load %in_v[%smidx_85] : memref<1511xi32>
        %sscale_85 = memref.load %in_v[%ssidx_85] : memref<1511xi32>
        %dilation_85 = arith.constant 9 : i32
        %padding_85 = arith.constant 36 : i32
        %ki0_85 = arith.constant 0 : i32
        %ki1_85 = arith.constant 1 : i32
        %ki2_85 = arith.constant 3 : i32
        %pad1_85 = arith.constant 0 : i32
        %outidx_85 = arith.constant 85 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_85, %smean_85, %sscale_85, %dilation_85, %padding_85, %ki0_85, %ki1_85, %ki2_85, %pad1_85, %out_v, %outidx_85) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 86: dilation=9 padding=36 kernel=(0,1,4) padding1=1
        %fidx_86 = arith.constant 86 : index
        %bidx_86 = arith.constant 337 : index
        %smidx_86 = arith.constant 757 : index
        %ssidx_86 = arith.constant 1177 : index
        %bias_86 = memref.load %in_v[%bidx_86] : memref<1511xi32>
        %smean_86 = memref.load %in_v[%smidx_86] : memref<1511xi32>
        %sscale_86 = memref.load %in_v[%ssidx_86] : memref<1511xi32>
        %dilation_86 = arith.constant 9 : i32
        %padding_86 = arith.constant 36 : i32
        %ki0_86 = arith.constant 0 : i32
        %ki1_86 = arith.constant 1 : i32
        %ki2_86 = arith.constant 4 : i32
        %pad1_86 = arith.constant 1 : i32
        %outidx_86 = arith.constant 86 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_86, %smean_86, %sscale_86, %dilation_86, %padding_86, %ki0_86, %ki1_86, %ki2_86, %pad1_86, %out_v, %outidx_86) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 87: dilation=9 padding=36 kernel=(0,1,5) padding1=0
        %fidx_87 = arith.constant 87 : index
        %bidx_87 = arith.constant 338 : index
        %smidx_87 = arith.constant 758 : index
        %ssidx_87 = arith.constant 1178 : index
        %bias_87 = memref.load %in_v[%bidx_87] : memref<1511xi32>
        %smean_87 = memref.load %in_v[%smidx_87] : memref<1511xi32>
        %sscale_87 = memref.load %in_v[%ssidx_87] : memref<1511xi32>
        %dilation_87 = arith.constant 9 : i32
        %padding_87 = arith.constant 36 : i32
        %ki0_87 = arith.constant 0 : i32
        %ki1_87 = arith.constant 1 : i32
        %ki2_87 = arith.constant 5 : i32
        %pad1_87 = arith.constant 0 : i32
        %outidx_87 = arith.constant 87 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_87, %smean_87, %sscale_87, %dilation_87, %padding_87, %ki0_87, %ki1_87, %ki2_87, %pad1_87, %out_v, %outidx_87) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 88: dilation=9 padding=36 kernel=(0,1,6) padding1=1
        %fidx_88 = arith.constant 88 : index
        %bidx_88 = arith.constant 339 : index
        %smidx_88 = arith.constant 759 : index
        %ssidx_88 = arith.constant 1179 : index
        %bias_88 = memref.load %in_v[%bidx_88] : memref<1511xi32>
        %smean_88 = memref.load %in_v[%smidx_88] : memref<1511xi32>
        %sscale_88 = memref.load %in_v[%ssidx_88] : memref<1511xi32>
        %dilation_88 = arith.constant 9 : i32
        %padding_88 = arith.constant 36 : i32
        %ki0_88 = arith.constant 0 : i32
        %ki1_88 = arith.constant 1 : i32
        %ki2_88 = arith.constant 6 : i32
        %pad1_88 = arith.constant 1 : i32
        %outidx_88 = arith.constant 88 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_88, %smean_88, %sscale_88, %dilation_88, %padding_88, %ki0_88, %ki1_88, %ki2_88, %pad1_88, %out_v, %outidx_88) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 89: dilation=9 padding=36 kernel=(0,1,7) padding1=0
        %fidx_89 = arith.constant 89 : index
        %bidx_89 = arith.constant 340 : index
        %smidx_89 = arith.constant 760 : index
        %ssidx_89 = arith.constant 1180 : index
        %bias_89 = memref.load %in_v[%bidx_89] : memref<1511xi32>
        %smean_89 = memref.load %in_v[%smidx_89] : memref<1511xi32>
        %sscale_89 = memref.load %in_v[%ssidx_89] : memref<1511xi32>
        %dilation_89 = arith.constant 9 : i32
        %padding_89 = arith.constant 36 : i32
        %ki0_89 = arith.constant 0 : i32
        %ki1_89 = arith.constant 1 : i32
        %ki2_89 = arith.constant 7 : i32
        %pad1_89 = arith.constant 0 : i32
        %outidx_89 = arith.constant 89 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_89, %smean_89, %sscale_89, %dilation_89, %padding_89, %ki0_89, %ki1_89, %ki2_89, %pad1_89, %out_v, %outidx_89) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 90: dilation=9 padding=36 kernel=(0,1,8) padding1=1
        %fidx_90 = arith.constant 90 : index
        %bidx_90 = arith.constant 341 : index
        %smidx_90 = arith.constant 761 : index
        %ssidx_90 = arith.constant 1181 : index
        %bias_90 = memref.load %in_v[%bidx_90] : memref<1511xi32>
        %smean_90 = memref.load %in_v[%smidx_90] : memref<1511xi32>
        %sscale_90 = memref.load %in_v[%ssidx_90] : memref<1511xi32>
        %dilation_90 = arith.constant 9 : i32
        %padding_90 = arith.constant 36 : i32
        %ki0_90 = arith.constant 0 : i32
        %ki1_90 = arith.constant 1 : i32
        %ki2_90 = arith.constant 8 : i32
        %pad1_90 = arith.constant 1 : i32
        %outidx_90 = arith.constant 90 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_90, %smean_90, %sscale_90, %dilation_90, %padding_90, %ki0_90, %ki1_90, %ki2_90, %pad1_90, %out_v, %outidx_90) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 91: dilation=9 padding=36 kernel=(0,2,3) padding1=0
        %fidx_91 = arith.constant 91 : index
        %bidx_91 = arith.constant 342 : index
        %smidx_91 = arith.constant 762 : index
        %ssidx_91 = arith.constant 1182 : index
        %bias_91 = memref.load %in_v[%bidx_91] : memref<1511xi32>
        %smean_91 = memref.load %in_v[%smidx_91] : memref<1511xi32>
        %sscale_91 = memref.load %in_v[%ssidx_91] : memref<1511xi32>
        %dilation_91 = arith.constant 9 : i32
        %padding_91 = arith.constant 36 : i32
        %ki0_91 = arith.constant 0 : i32
        %ki1_91 = arith.constant 2 : i32
        %ki2_91 = arith.constant 3 : i32
        %pad1_91 = arith.constant 0 : i32
        %outidx_91 = arith.constant 91 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_91, %smean_91, %sscale_91, %dilation_91, %padding_91, %ki0_91, %ki1_91, %ki2_91, %pad1_91, %out_v, %outidx_91) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 92: dilation=9 padding=36 kernel=(0,2,4) padding1=1
        %fidx_92 = arith.constant 92 : index
        %bidx_92 = arith.constant 343 : index
        %smidx_92 = arith.constant 763 : index
        %ssidx_92 = arith.constant 1183 : index
        %bias_92 = memref.load %in_v[%bidx_92] : memref<1511xi32>
        %smean_92 = memref.load %in_v[%smidx_92] : memref<1511xi32>
        %sscale_92 = memref.load %in_v[%ssidx_92] : memref<1511xi32>
        %dilation_92 = arith.constant 9 : i32
        %padding_92 = arith.constant 36 : i32
        %ki0_92 = arith.constant 0 : i32
        %ki1_92 = arith.constant 2 : i32
        %ki2_92 = arith.constant 4 : i32
        %pad1_92 = arith.constant 1 : i32
        %outidx_92 = arith.constant 92 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_92, %smean_92, %sscale_92, %dilation_92, %padding_92, %ki0_92, %ki1_92, %ki2_92, %pad1_92, %out_v, %outidx_92) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 93: dilation=9 padding=36 kernel=(0,2,5) padding1=0
        %fidx_93 = arith.constant 93 : index
        %bidx_93 = arith.constant 344 : index
        %smidx_93 = arith.constant 764 : index
        %ssidx_93 = arith.constant 1184 : index
        %bias_93 = memref.load %in_v[%bidx_93] : memref<1511xi32>
        %smean_93 = memref.load %in_v[%smidx_93] : memref<1511xi32>
        %sscale_93 = memref.load %in_v[%ssidx_93] : memref<1511xi32>
        %dilation_93 = arith.constant 9 : i32
        %padding_93 = arith.constant 36 : i32
        %ki0_93 = arith.constant 0 : i32
        %ki1_93 = arith.constant 2 : i32
        %ki2_93 = arith.constant 5 : i32
        %pad1_93 = arith.constant 0 : i32
        %outidx_93 = arith.constant 93 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_93, %smean_93, %sscale_93, %dilation_93, %padding_93, %ki0_93, %ki1_93, %ki2_93, %pad1_93, %out_v, %outidx_93) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 94: dilation=9 padding=36 kernel=(0,2,6) padding1=1
        %fidx_94 = arith.constant 94 : index
        %bidx_94 = arith.constant 345 : index
        %smidx_94 = arith.constant 765 : index
        %ssidx_94 = arith.constant 1185 : index
        %bias_94 = memref.load %in_v[%bidx_94] : memref<1511xi32>
        %smean_94 = memref.load %in_v[%smidx_94] : memref<1511xi32>
        %sscale_94 = memref.load %in_v[%ssidx_94] : memref<1511xi32>
        %dilation_94 = arith.constant 9 : i32
        %padding_94 = arith.constant 36 : i32
        %ki0_94 = arith.constant 0 : i32
        %ki1_94 = arith.constant 2 : i32
        %ki2_94 = arith.constant 6 : i32
        %pad1_94 = arith.constant 1 : i32
        %outidx_94 = arith.constant 94 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_94, %smean_94, %sscale_94, %dilation_94, %padding_94, %ki0_94, %ki1_94, %ki2_94, %pad1_94, %out_v, %outidx_94) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 95: dilation=9 padding=36 kernel=(0,2,7) padding1=0
        %fidx_95 = arith.constant 95 : index
        %bidx_95 = arith.constant 346 : index
        %smidx_95 = arith.constant 766 : index
        %ssidx_95 = arith.constant 1186 : index
        %bias_95 = memref.load %in_v[%bidx_95] : memref<1511xi32>
        %smean_95 = memref.load %in_v[%smidx_95] : memref<1511xi32>
        %sscale_95 = memref.load %in_v[%ssidx_95] : memref<1511xi32>
        %dilation_95 = arith.constant 9 : i32
        %padding_95 = arith.constant 36 : i32
        %ki0_95 = arith.constant 0 : i32
        %ki1_95 = arith.constant 2 : i32
        %ki2_95 = arith.constant 7 : i32
        %pad1_95 = arith.constant 0 : i32
        %outidx_95 = arith.constant 95 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_95, %smean_95, %sscale_95, %dilation_95, %padding_95, %ki0_95, %ki1_95, %ki2_95, %pad1_95, %out_v, %outidx_95) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 96: dilation=9 padding=36 kernel=(0,2,8) padding1=1
        %fidx_96 = arith.constant 96 : index
        %bidx_96 = arith.constant 347 : index
        %smidx_96 = arith.constant 767 : index
        %ssidx_96 = arith.constant 1187 : index
        %bias_96 = memref.load %in_v[%bidx_96] : memref<1511xi32>
        %smean_96 = memref.load %in_v[%smidx_96] : memref<1511xi32>
        %sscale_96 = memref.load %in_v[%ssidx_96] : memref<1511xi32>
        %dilation_96 = arith.constant 9 : i32
        %padding_96 = arith.constant 36 : i32
        %ki0_96 = arith.constant 0 : i32
        %ki1_96 = arith.constant 2 : i32
        %ki2_96 = arith.constant 8 : i32
        %pad1_96 = arith.constant 1 : i32
        %outidx_96 = arith.constant 96 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_96, %smean_96, %sscale_96, %dilation_96, %padding_96, %ki0_96, %ki1_96, %ki2_96, %pad1_96, %out_v, %outidx_96) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 97: dilation=9 padding=36 kernel=(0,3,4) padding1=0
        %fidx_97 = arith.constant 97 : index
        %bidx_97 = arith.constant 348 : index
        %smidx_97 = arith.constant 768 : index
        %ssidx_97 = arith.constant 1188 : index
        %bias_97 = memref.load %in_v[%bidx_97] : memref<1511xi32>
        %smean_97 = memref.load %in_v[%smidx_97] : memref<1511xi32>
        %sscale_97 = memref.load %in_v[%ssidx_97] : memref<1511xi32>
        %dilation_97 = arith.constant 9 : i32
        %padding_97 = arith.constant 36 : i32
        %ki0_97 = arith.constant 0 : i32
        %ki1_97 = arith.constant 3 : i32
        %ki2_97 = arith.constant 4 : i32
        %pad1_97 = arith.constant 0 : i32
        %outidx_97 = arith.constant 97 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_97, %smean_97, %sscale_97, %dilation_97, %padding_97, %ki0_97, %ki1_97, %ki2_97, %pad1_97, %out_v, %outidx_97) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 98: dilation=9 padding=36 kernel=(0,3,5) padding1=1
        %fidx_98 = arith.constant 98 : index
        %bidx_98 = arith.constant 349 : index
        %smidx_98 = arith.constant 769 : index
        %ssidx_98 = arith.constant 1189 : index
        %bias_98 = memref.load %in_v[%bidx_98] : memref<1511xi32>
        %smean_98 = memref.load %in_v[%smidx_98] : memref<1511xi32>
        %sscale_98 = memref.load %in_v[%ssidx_98] : memref<1511xi32>
        %dilation_98 = arith.constant 9 : i32
        %padding_98 = arith.constant 36 : i32
        %ki0_98 = arith.constant 0 : i32
        %ki1_98 = arith.constant 3 : i32
        %ki2_98 = arith.constant 5 : i32
        %pad1_98 = arith.constant 1 : i32
        %outidx_98 = arith.constant 98 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_98, %smean_98, %sscale_98, %dilation_98, %padding_98, %ki0_98, %ki1_98, %ki2_98, %pad1_98, %out_v, %outidx_98) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 99: dilation=9 padding=36 kernel=(0,3,6) padding1=0
        %fidx_99 = arith.constant 99 : index
        %bidx_99 = arith.constant 350 : index
        %smidx_99 = arith.constant 770 : index
        %ssidx_99 = arith.constant 1190 : index
        %bias_99 = memref.load %in_v[%bidx_99] : memref<1511xi32>
        %smean_99 = memref.load %in_v[%smidx_99] : memref<1511xi32>
        %sscale_99 = memref.load %in_v[%ssidx_99] : memref<1511xi32>
        %dilation_99 = arith.constant 9 : i32
        %padding_99 = arith.constant 36 : i32
        %ki0_99 = arith.constant 0 : i32
        %ki1_99 = arith.constant 3 : i32
        %ki2_99 = arith.constant 6 : i32
        %pad1_99 = arith.constant 0 : i32
        %outidx_99 = arith.constant 99 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_99, %smean_99, %sscale_99, %dilation_99, %padding_99, %ki0_99, %ki1_99, %ki2_99, %pad1_99, %out_v, %outidx_99) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 100: dilation=9 padding=36 kernel=(0,3,7) padding1=1
        %fidx_100 = arith.constant 100 : index
        %bidx_100 = arith.constant 351 : index
        %smidx_100 = arith.constant 771 : index
        %ssidx_100 = arith.constant 1191 : index
        %bias_100 = memref.load %in_v[%bidx_100] : memref<1511xi32>
        %smean_100 = memref.load %in_v[%smidx_100] : memref<1511xi32>
        %sscale_100 = memref.load %in_v[%ssidx_100] : memref<1511xi32>
        %dilation_100 = arith.constant 9 : i32
        %padding_100 = arith.constant 36 : i32
        %ki0_100 = arith.constant 0 : i32
        %ki1_100 = arith.constant 3 : i32
        %ki2_100 = arith.constant 7 : i32
        %pad1_100 = arith.constant 1 : i32
        %outidx_100 = arith.constant 100 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_100, %smean_100, %sscale_100, %dilation_100, %padding_100, %ki0_100, %ki1_100, %ki2_100, %pad1_100, %out_v, %outidx_100) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 101: dilation=9 padding=36 kernel=(0,3,8) padding1=0
        %fidx_101 = arith.constant 101 : index
        %bidx_101 = arith.constant 352 : index
        %smidx_101 = arith.constant 772 : index
        %ssidx_101 = arith.constant 1192 : index
        %bias_101 = memref.load %in_v[%bidx_101] : memref<1511xi32>
        %smean_101 = memref.load %in_v[%smidx_101] : memref<1511xi32>
        %sscale_101 = memref.load %in_v[%ssidx_101] : memref<1511xi32>
        %dilation_101 = arith.constant 9 : i32
        %padding_101 = arith.constant 36 : i32
        %ki0_101 = arith.constant 0 : i32
        %ki1_101 = arith.constant 3 : i32
        %ki2_101 = arith.constant 8 : i32
        %pad1_101 = arith.constant 0 : i32
        %outidx_101 = arith.constant 101 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_101, %smean_101, %sscale_101, %dilation_101, %padding_101, %ki0_101, %ki1_101, %ki2_101, %pad1_101, %out_v, %outidx_101) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 102: dilation=9 padding=36 kernel=(0,4,5) padding1=1
        %fidx_102 = arith.constant 102 : index
        %bidx_102 = arith.constant 353 : index
        %smidx_102 = arith.constant 773 : index
        %ssidx_102 = arith.constant 1193 : index
        %bias_102 = memref.load %in_v[%bidx_102] : memref<1511xi32>
        %smean_102 = memref.load %in_v[%smidx_102] : memref<1511xi32>
        %sscale_102 = memref.load %in_v[%ssidx_102] : memref<1511xi32>
        %dilation_102 = arith.constant 9 : i32
        %padding_102 = arith.constant 36 : i32
        %ki0_102 = arith.constant 0 : i32
        %ki1_102 = arith.constant 4 : i32
        %ki2_102 = arith.constant 5 : i32
        %pad1_102 = arith.constant 1 : i32
        %outidx_102 = arith.constant 102 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_102, %smean_102, %sscale_102, %dilation_102, %padding_102, %ki0_102, %ki1_102, %ki2_102, %pad1_102, %out_v, %outidx_102) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 103: dilation=9 padding=36 kernel=(0,4,6) padding1=0
        %fidx_103 = arith.constant 103 : index
        %bidx_103 = arith.constant 354 : index
        %smidx_103 = arith.constant 774 : index
        %ssidx_103 = arith.constant 1194 : index
        %bias_103 = memref.load %in_v[%bidx_103] : memref<1511xi32>
        %smean_103 = memref.load %in_v[%smidx_103] : memref<1511xi32>
        %sscale_103 = memref.load %in_v[%ssidx_103] : memref<1511xi32>
        %dilation_103 = arith.constant 9 : i32
        %padding_103 = arith.constant 36 : i32
        %ki0_103 = arith.constant 0 : i32
        %ki1_103 = arith.constant 4 : i32
        %ki2_103 = arith.constant 6 : i32
        %pad1_103 = arith.constant 0 : i32
        %outidx_103 = arith.constant 103 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_103, %smean_103, %sscale_103, %dilation_103, %padding_103, %ki0_103, %ki1_103, %ki2_103, %pad1_103, %out_v, %outidx_103) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 104: dilation=9 padding=36 kernel=(0,4,7) padding1=1
        %fidx_104 = arith.constant 104 : index
        %bidx_104 = arith.constant 355 : index
        %smidx_104 = arith.constant 775 : index
        %ssidx_104 = arith.constant 1195 : index
        %bias_104 = memref.load %in_v[%bidx_104] : memref<1511xi32>
        %smean_104 = memref.load %in_v[%smidx_104] : memref<1511xi32>
        %sscale_104 = memref.load %in_v[%ssidx_104] : memref<1511xi32>
        %dilation_104 = arith.constant 9 : i32
        %padding_104 = arith.constant 36 : i32
        %ki0_104 = arith.constant 0 : i32
        %ki1_104 = arith.constant 4 : i32
        %ki2_104 = arith.constant 7 : i32
        %pad1_104 = arith.constant 1 : i32
        %outidx_104 = arith.constant 104 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_104, %smean_104, %sscale_104, %dilation_104, %padding_104, %ki0_104, %ki1_104, %ki2_104, %pad1_104, %out_v, %outidx_104) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 105: dilation=9 padding=36 kernel=(0,4,8) padding1=0
        %fidx_105 = arith.constant 105 : index
        %bidx_105 = arith.constant 356 : index
        %smidx_105 = arith.constant 776 : index
        %ssidx_105 = arith.constant 1196 : index
        %bias_105 = memref.load %in_v[%bidx_105] : memref<1511xi32>
        %smean_105 = memref.load %in_v[%smidx_105] : memref<1511xi32>
        %sscale_105 = memref.load %in_v[%ssidx_105] : memref<1511xi32>
        %dilation_105 = arith.constant 9 : i32
        %padding_105 = arith.constant 36 : i32
        %ki0_105 = arith.constant 0 : i32
        %ki1_105 = arith.constant 4 : i32
        %ki2_105 = arith.constant 8 : i32
        %pad1_105 = arith.constant 0 : i32
        %outidx_105 = arith.constant 105 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_105, %smean_105, %sscale_105, %dilation_105, %padding_105, %ki0_105, %ki1_105, %ki2_105, %pad1_105, %out_v, %outidx_105) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 106: dilation=9 padding=36 kernel=(0,5,6) padding1=1
        %fidx_106 = arith.constant 106 : index
        %bidx_106 = arith.constant 357 : index
        %smidx_106 = arith.constant 777 : index
        %ssidx_106 = arith.constant 1197 : index
        %bias_106 = memref.load %in_v[%bidx_106] : memref<1511xi32>
        %smean_106 = memref.load %in_v[%smidx_106] : memref<1511xi32>
        %sscale_106 = memref.load %in_v[%ssidx_106] : memref<1511xi32>
        %dilation_106 = arith.constant 9 : i32
        %padding_106 = arith.constant 36 : i32
        %ki0_106 = arith.constant 0 : i32
        %ki1_106 = arith.constant 5 : i32
        %ki2_106 = arith.constant 6 : i32
        %pad1_106 = arith.constant 1 : i32
        %outidx_106 = arith.constant 106 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_106, %smean_106, %sscale_106, %dilation_106, %padding_106, %ki0_106, %ki1_106, %ki2_106, %pad1_106, %out_v, %outidx_106) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 107: dilation=9 padding=36 kernel=(0,5,7) padding1=0
        %fidx_107 = arith.constant 107 : index
        %bidx_107 = arith.constant 358 : index
        %smidx_107 = arith.constant 778 : index
        %ssidx_107 = arith.constant 1198 : index
        %bias_107 = memref.load %in_v[%bidx_107] : memref<1511xi32>
        %smean_107 = memref.load %in_v[%smidx_107] : memref<1511xi32>
        %sscale_107 = memref.load %in_v[%ssidx_107] : memref<1511xi32>
        %dilation_107 = arith.constant 9 : i32
        %padding_107 = arith.constant 36 : i32
        %ki0_107 = arith.constant 0 : i32
        %ki1_107 = arith.constant 5 : i32
        %ki2_107 = arith.constant 7 : i32
        %pad1_107 = arith.constant 0 : i32
        %outidx_107 = arith.constant 107 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_107, %smean_107, %sscale_107, %dilation_107, %padding_107, %ki0_107, %ki1_107, %ki2_107, %pad1_107, %out_v, %outidx_107) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 108: dilation=9 padding=36 kernel=(0,5,8) padding1=1
        %fidx_108 = arith.constant 108 : index
        %bidx_108 = arith.constant 359 : index
        %smidx_108 = arith.constant 779 : index
        %ssidx_108 = arith.constant 1199 : index
        %bias_108 = memref.load %in_v[%bidx_108] : memref<1511xi32>
        %smean_108 = memref.load %in_v[%smidx_108] : memref<1511xi32>
        %sscale_108 = memref.load %in_v[%ssidx_108] : memref<1511xi32>
        %dilation_108 = arith.constant 9 : i32
        %padding_108 = arith.constant 36 : i32
        %ki0_108 = arith.constant 0 : i32
        %ki1_108 = arith.constant 5 : i32
        %ki2_108 = arith.constant 8 : i32
        %pad1_108 = arith.constant 1 : i32
        %outidx_108 = arith.constant 108 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_108, %smean_108, %sscale_108, %dilation_108, %padding_108, %ki0_108, %ki1_108, %ki2_108, %pad1_108, %out_v, %outidx_108) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 109: dilation=9 padding=36 kernel=(0,6,7) padding1=0
        %fidx_109 = arith.constant 109 : index
        %bidx_109 = arith.constant 360 : index
        %smidx_109 = arith.constant 780 : index
        %ssidx_109 = arith.constant 1200 : index
        %bias_109 = memref.load %in_v[%bidx_109] : memref<1511xi32>
        %smean_109 = memref.load %in_v[%smidx_109] : memref<1511xi32>
        %sscale_109 = memref.load %in_v[%ssidx_109] : memref<1511xi32>
        %dilation_109 = arith.constant 9 : i32
        %padding_109 = arith.constant 36 : i32
        %ki0_109 = arith.constant 0 : i32
        %ki1_109 = arith.constant 6 : i32
        %ki2_109 = arith.constant 7 : i32
        %pad1_109 = arith.constant 0 : i32
        %outidx_109 = arith.constant 109 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_109, %smean_109, %sscale_109, %dilation_109, %padding_109, %ki0_109, %ki1_109, %ki2_109, %pad1_109, %out_v, %outidx_109) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 110: dilation=9 padding=36 kernel=(0,6,8) padding1=1
        %fidx_110 = arith.constant 110 : index
        %bidx_110 = arith.constant 361 : index
        %smidx_110 = arith.constant 781 : index
        %ssidx_110 = arith.constant 1201 : index
        %bias_110 = memref.load %in_v[%bidx_110] : memref<1511xi32>
        %smean_110 = memref.load %in_v[%smidx_110] : memref<1511xi32>
        %sscale_110 = memref.load %in_v[%ssidx_110] : memref<1511xi32>
        %dilation_110 = arith.constant 9 : i32
        %padding_110 = arith.constant 36 : i32
        %ki0_110 = arith.constant 0 : i32
        %ki1_110 = arith.constant 6 : i32
        %ki2_110 = arith.constant 8 : i32
        %pad1_110 = arith.constant 1 : i32
        %outidx_110 = arith.constant 110 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_110, %smean_110, %sscale_110, %dilation_110, %padding_110, %ki0_110, %ki1_110, %ki2_110, %pad1_110, %out_v, %outidx_110) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 111: dilation=9 padding=36 kernel=(0,7,8) padding1=0
        %fidx_111 = arith.constant 111 : index
        %bidx_111 = arith.constant 362 : index
        %smidx_111 = arith.constant 782 : index
        %ssidx_111 = arith.constant 1202 : index
        %bias_111 = memref.load %in_v[%bidx_111] : memref<1511xi32>
        %smean_111 = memref.load %in_v[%smidx_111] : memref<1511xi32>
        %sscale_111 = memref.load %in_v[%ssidx_111] : memref<1511xi32>
        %dilation_111 = arith.constant 9 : i32
        %padding_111 = arith.constant 36 : i32
        %ki0_111 = arith.constant 0 : i32
        %ki1_111 = arith.constant 7 : i32
        %ki2_111 = arith.constant 8 : i32
        %pad1_111 = arith.constant 0 : i32
        %outidx_111 = arith.constant 111 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_111, %smean_111, %sscale_111, %dilation_111, %padding_111, %ki0_111, %ki1_111, %ki2_111, %pad1_111, %out_v, %outidx_111) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 112: dilation=9 padding=36 kernel=(1,2,3) padding1=1
        %fidx_112 = arith.constant 112 : index
        %bidx_112 = arith.constant 363 : index
        %smidx_112 = arith.constant 783 : index
        %ssidx_112 = arith.constant 1203 : index
        %bias_112 = memref.load %in_v[%bidx_112] : memref<1511xi32>
        %smean_112 = memref.load %in_v[%smidx_112] : memref<1511xi32>
        %sscale_112 = memref.load %in_v[%ssidx_112] : memref<1511xi32>
        %dilation_112 = arith.constant 9 : i32
        %padding_112 = arith.constant 36 : i32
        %ki0_112 = arith.constant 1 : i32
        %ki1_112 = arith.constant 2 : i32
        %ki2_112 = arith.constant 3 : i32
        %pad1_112 = arith.constant 1 : i32
        %outidx_112 = arith.constant 112 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_112, %smean_112, %sscale_112, %dilation_112, %padding_112, %ki0_112, %ki1_112, %ki2_112, %pad1_112, %out_v, %outidx_112) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 113: dilation=9 padding=36 kernel=(1,2,4) padding1=0
        %fidx_113 = arith.constant 113 : index
        %bidx_113 = arith.constant 364 : index
        %smidx_113 = arith.constant 784 : index
        %ssidx_113 = arith.constant 1204 : index
        %bias_113 = memref.load %in_v[%bidx_113] : memref<1511xi32>
        %smean_113 = memref.load %in_v[%smidx_113] : memref<1511xi32>
        %sscale_113 = memref.load %in_v[%ssidx_113] : memref<1511xi32>
        %dilation_113 = arith.constant 9 : i32
        %padding_113 = arith.constant 36 : i32
        %ki0_113 = arith.constant 1 : i32
        %ki1_113 = arith.constant 2 : i32
        %ki2_113 = arith.constant 4 : i32
        %pad1_113 = arith.constant 0 : i32
        %outidx_113 = arith.constant 113 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_113, %smean_113, %sscale_113, %dilation_113, %padding_113, %ki0_113, %ki1_113, %ki2_113, %pad1_113, %out_v, %outidx_113) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 114: dilation=9 padding=36 kernel=(1,2,5) padding1=1
        %fidx_114 = arith.constant 114 : index
        %bidx_114 = arith.constant 365 : index
        %smidx_114 = arith.constant 785 : index
        %ssidx_114 = arith.constant 1205 : index
        %bias_114 = memref.load %in_v[%bidx_114] : memref<1511xi32>
        %smean_114 = memref.load %in_v[%smidx_114] : memref<1511xi32>
        %sscale_114 = memref.load %in_v[%ssidx_114] : memref<1511xi32>
        %dilation_114 = arith.constant 9 : i32
        %padding_114 = arith.constant 36 : i32
        %ki0_114 = arith.constant 1 : i32
        %ki1_114 = arith.constant 2 : i32
        %ki2_114 = arith.constant 5 : i32
        %pad1_114 = arith.constant 1 : i32
        %outidx_114 = arith.constant 114 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_114, %smean_114, %sscale_114, %dilation_114, %padding_114, %ki0_114, %ki1_114, %ki2_114, %pad1_114, %out_v, %outidx_114) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 115: dilation=9 padding=36 kernel=(1,2,6) padding1=0
        %fidx_115 = arith.constant 115 : index
        %bidx_115 = arith.constant 366 : index
        %smidx_115 = arith.constant 786 : index
        %ssidx_115 = arith.constant 1206 : index
        %bias_115 = memref.load %in_v[%bidx_115] : memref<1511xi32>
        %smean_115 = memref.load %in_v[%smidx_115] : memref<1511xi32>
        %sscale_115 = memref.load %in_v[%ssidx_115] : memref<1511xi32>
        %dilation_115 = arith.constant 9 : i32
        %padding_115 = arith.constant 36 : i32
        %ki0_115 = arith.constant 1 : i32
        %ki1_115 = arith.constant 2 : i32
        %ki2_115 = arith.constant 6 : i32
        %pad1_115 = arith.constant 0 : i32
        %outidx_115 = arith.constant 115 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_115, %smean_115, %sscale_115, %dilation_115, %padding_115, %ki0_115, %ki1_115, %ki2_115, %pad1_115, %out_v, %outidx_115) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 116: dilation=9 padding=36 kernel=(1,2,7) padding1=1
        %fidx_116 = arith.constant 116 : index
        %bidx_116 = arith.constant 367 : index
        %smidx_116 = arith.constant 787 : index
        %ssidx_116 = arith.constant 1207 : index
        %bias_116 = memref.load %in_v[%bidx_116] : memref<1511xi32>
        %smean_116 = memref.load %in_v[%smidx_116] : memref<1511xi32>
        %sscale_116 = memref.load %in_v[%ssidx_116] : memref<1511xi32>
        %dilation_116 = arith.constant 9 : i32
        %padding_116 = arith.constant 36 : i32
        %ki0_116 = arith.constant 1 : i32
        %ki1_116 = arith.constant 2 : i32
        %ki2_116 = arith.constant 7 : i32
        %pad1_116 = arith.constant 1 : i32
        %outidx_116 = arith.constant 116 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_116, %smean_116, %sscale_116, %dilation_116, %padding_116, %ki0_116, %ki1_116, %ki2_116, %pad1_116, %out_v, %outidx_116) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 117: dilation=9 padding=36 kernel=(1,2,8) padding1=0
        %fidx_117 = arith.constant 117 : index
        %bidx_117 = arith.constant 368 : index
        %smidx_117 = arith.constant 788 : index
        %ssidx_117 = arith.constant 1208 : index
        %bias_117 = memref.load %in_v[%bidx_117] : memref<1511xi32>
        %smean_117 = memref.load %in_v[%smidx_117] : memref<1511xi32>
        %sscale_117 = memref.load %in_v[%ssidx_117] : memref<1511xi32>
        %dilation_117 = arith.constant 9 : i32
        %padding_117 = arith.constant 36 : i32
        %ki0_117 = arith.constant 1 : i32
        %ki1_117 = arith.constant 2 : i32
        %ki2_117 = arith.constant 8 : i32
        %pad1_117 = arith.constant 0 : i32
        %outidx_117 = arith.constant 117 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_117, %smean_117, %sscale_117, %dilation_117, %padding_117, %ki0_117, %ki1_117, %ki2_117, %pad1_117, %out_v, %outidx_117) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 118: dilation=9 padding=36 kernel=(1,3,4) padding1=1
        %fidx_118 = arith.constant 118 : index
        %bidx_118 = arith.constant 369 : index
        %smidx_118 = arith.constant 789 : index
        %ssidx_118 = arith.constant 1209 : index
        %bias_118 = memref.load %in_v[%bidx_118] : memref<1511xi32>
        %smean_118 = memref.load %in_v[%smidx_118] : memref<1511xi32>
        %sscale_118 = memref.load %in_v[%ssidx_118] : memref<1511xi32>
        %dilation_118 = arith.constant 9 : i32
        %padding_118 = arith.constant 36 : i32
        %ki0_118 = arith.constant 1 : i32
        %ki1_118 = arith.constant 3 : i32
        %ki2_118 = arith.constant 4 : i32
        %pad1_118 = arith.constant 1 : i32
        %outidx_118 = arith.constant 118 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_118, %smean_118, %sscale_118, %dilation_118, %padding_118, %ki0_118, %ki1_118, %ki2_118, %pad1_118, %out_v, %outidx_118) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 119: dilation=9 padding=36 kernel=(1,3,5) padding1=0
        %fidx_119 = arith.constant 119 : index
        %bidx_119 = arith.constant 370 : index
        %smidx_119 = arith.constant 790 : index
        %ssidx_119 = arith.constant 1210 : index
        %bias_119 = memref.load %in_v[%bidx_119] : memref<1511xi32>
        %smean_119 = memref.load %in_v[%smidx_119] : memref<1511xi32>
        %sscale_119 = memref.load %in_v[%ssidx_119] : memref<1511xi32>
        %dilation_119 = arith.constant 9 : i32
        %padding_119 = arith.constant 36 : i32
        %ki0_119 = arith.constant 1 : i32
        %ki1_119 = arith.constant 3 : i32
        %ki2_119 = arith.constant 5 : i32
        %pad1_119 = arith.constant 0 : i32
        %outidx_119 = arith.constant 119 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_119, %smean_119, %sscale_119, %dilation_119, %padding_119, %ki0_119, %ki1_119, %ki2_119, %pad1_119, %out_v, %outidx_119) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 120: dilation=9 padding=36 kernel=(1,3,6) padding1=1
        %fidx_120 = arith.constant 120 : index
        %bidx_120 = arith.constant 371 : index
        %smidx_120 = arith.constant 791 : index
        %ssidx_120 = arith.constant 1211 : index
        %bias_120 = memref.load %in_v[%bidx_120] : memref<1511xi32>
        %smean_120 = memref.load %in_v[%smidx_120] : memref<1511xi32>
        %sscale_120 = memref.load %in_v[%ssidx_120] : memref<1511xi32>
        %dilation_120 = arith.constant 9 : i32
        %padding_120 = arith.constant 36 : i32
        %ki0_120 = arith.constant 1 : i32
        %ki1_120 = arith.constant 3 : i32
        %ki2_120 = arith.constant 6 : i32
        %pad1_120 = arith.constant 1 : i32
        %outidx_120 = arith.constant 120 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_120, %smean_120, %sscale_120, %dilation_120, %padding_120, %ki0_120, %ki1_120, %ki2_120, %pad1_120, %out_v, %outidx_120) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 121: dilation=9 padding=36 kernel=(1,3,7) padding1=0
        %fidx_121 = arith.constant 121 : index
        %bidx_121 = arith.constant 372 : index
        %smidx_121 = arith.constant 792 : index
        %ssidx_121 = arith.constant 1212 : index
        %bias_121 = memref.load %in_v[%bidx_121] : memref<1511xi32>
        %smean_121 = memref.load %in_v[%smidx_121] : memref<1511xi32>
        %sscale_121 = memref.load %in_v[%ssidx_121] : memref<1511xi32>
        %dilation_121 = arith.constant 9 : i32
        %padding_121 = arith.constant 36 : i32
        %ki0_121 = arith.constant 1 : i32
        %ki1_121 = arith.constant 3 : i32
        %ki2_121 = arith.constant 7 : i32
        %pad1_121 = arith.constant 0 : i32
        %outidx_121 = arith.constant 121 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_121, %smean_121, %sscale_121, %dilation_121, %padding_121, %ki0_121, %ki1_121, %ki2_121, %pad1_121, %out_v, %outidx_121) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 122: dilation=9 padding=36 kernel=(1,3,8) padding1=1
        %fidx_122 = arith.constant 122 : index
        %bidx_122 = arith.constant 373 : index
        %smidx_122 = arith.constant 793 : index
        %ssidx_122 = arith.constant 1213 : index
        %bias_122 = memref.load %in_v[%bidx_122] : memref<1511xi32>
        %smean_122 = memref.load %in_v[%smidx_122] : memref<1511xi32>
        %sscale_122 = memref.load %in_v[%ssidx_122] : memref<1511xi32>
        %dilation_122 = arith.constant 9 : i32
        %padding_122 = arith.constant 36 : i32
        %ki0_122 = arith.constant 1 : i32
        %ki1_122 = arith.constant 3 : i32
        %ki2_122 = arith.constant 8 : i32
        %pad1_122 = arith.constant 1 : i32
        %outidx_122 = arith.constant 122 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_122, %smean_122, %sscale_122, %dilation_122, %padding_122, %ki0_122, %ki1_122, %ki2_122, %pad1_122, %out_v, %outidx_122) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 123: dilation=9 padding=36 kernel=(1,4,5) padding1=0
        %fidx_123 = arith.constant 123 : index
        %bidx_123 = arith.constant 374 : index
        %smidx_123 = arith.constant 794 : index
        %ssidx_123 = arith.constant 1214 : index
        %bias_123 = memref.load %in_v[%bidx_123] : memref<1511xi32>
        %smean_123 = memref.load %in_v[%smidx_123] : memref<1511xi32>
        %sscale_123 = memref.load %in_v[%ssidx_123] : memref<1511xi32>
        %dilation_123 = arith.constant 9 : i32
        %padding_123 = arith.constant 36 : i32
        %ki0_123 = arith.constant 1 : i32
        %ki1_123 = arith.constant 4 : i32
        %ki2_123 = arith.constant 5 : i32
        %pad1_123 = arith.constant 0 : i32
        %outidx_123 = arith.constant 123 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_123, %smean_123, %sscale_123, %dilation_123, %padding_123, %ki0_123, %ki1_123, %ki2_123, %pad1_123, %out_v, %outidx_123) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 124: dilation=9 padding=36 kernel=(1,4,6) padding1=1
        %fidx_124 = arith.constant 124 : index
        %bidx_124 = arith.constant 375 : index
        %smidx_124 = arith.constant 795 : index
        %ssidx_124 = arith.constant 1215 : index
        %bias_124 = memref.load %in_v[%bidx_124] : memref<1511xi32>
        %smean_124 = memref.load %in_v[%smidx_124] : memref<1511xi32>
        %sscale_124 = memref.load %in_v[%ssidx_124] : memref<1511xi32>
        %dilation_124 = arith.constant 9 : i32
        %padding_124 = arith.constant 36 : i32
        %ki0_124 = arith.constant 1 : i32
        %ki1_124 = arith.constant 4 : i32
        %ki2_124 = arith.constant 6 : i32
        %pad1_124 = arith.constant 1 : i32
        %outidx_124 = arith.constant 124 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_124, %smean_124, %sscale_124, %dilation_124, %padding_124, %ki0_124, %ki1_124, %ki2_124, %pad1_124, %out_v, %outidx_124) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 125: dilation=9 padding=36 kernel=(1,4,7) padding1=0
        %fidx_125 = arith.constant 125 : index
        %bidx_125 = arith.constant 376 : index
        %smidx_125 = arith.constant 796 : index
        %ssidx_125 = arith.constant 1216 : index
        %bias_125 = memref.load %in_v[%bidx_125] : memref<1511xi32>
        %smean_125 = memref.load %in_v[%smidx_125] : memref<1511xi32>
        %sscale_125 = memref.load %in_v[%ssidx_125] : memref<1511xi32>
        %dilation_125 = arith.constant 9 : i32
        %padding_125 = arith.constant 36 : i32
        %ki0_125 = arith.constant 1 : i32
        %ki1_125 = arith.constant 4 : i32
        %ki2_125 = arith.constant 7 : i32
        %pad1_125 = arith.constant 0 : i32
        %outidx_125 = arith.constant 125 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_125, %smean_125, %sscale_125, %dilation_125, %padding_125, %ki0_125, %ki1_125, %ki2_125, %pad1_125, %out_v, %outidx_125) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 126: dilation=9 padding=36 kernel=(1,4,8) padding1=1
        %fidx_126 = arith.constant 126 : index
        %bidx_126 = arith.constant 377 : index
        %smidx_126 = arith.constant 797 : index
        %ssidx_126 = arith.constant 1217 : index
        %bias_126 = memref.load %in_v[%bidx_126] : memref<1511xi32>
        %smean_126 = memref.load %in_v[%smidx_126] : memref<1511xi32>
        %sscale_126 = memref.load %in_v[%ssidx_126] : memref<1511xi32>
        %dilation_126 = arith.constant 9 : i32
        %padding_126 = arith.constant 36 : i32
        %ki0_126 = arith.constant 1 : i32
        %ki1_126 = arith.constant 4 : i32
        %ki2_126 = arith.constant 8 : i32
        %pad1_126 = arith.constant 1 : i32
        %outidx_126 = arith.constant 126 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_126, %smean_126, %sscale_126, %dilation_126, %padding_126, %ki0_126, %ki1_126, %ki2_126, %pad1_126, %out_v, %outidx_126) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 127: dilation=9 padding=36 kernel=(1,5,6) padding1=0
        %fidx_127 = arith.constant 127 : index
        %bidx_127 = arith.constant 378 : index
        %smidx_127 = arith.constant 798 : index
        %ssidx_127 = arith.constant 1218 : index
        %bias_127 = memref.load %in_v[%bidx_127] : memref<1511xi32>
        %smean_127 = memref.load %in_v[%smidx_127] : memref<1511xi32>
        %sscale_127 = memref.load %in_v[%ssidx_127] : memref<1511xi32>
        %dilation_127 = arith.constant 9 : i32
        %padding_127 = arith.constant 36 : i32
        %ki0_127 = arith.constant 1 : i32
        %ki1_127 = arith.constant 5 : i32
        %ki2_127 = arith.constant 6 : i32
        %pad1_127 = arith.constant 0 : i32
        %outidx_127 = arith.constant 127 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_127, %smean_127, %sscale_127, %dilation_127, %padding_127, %ki0_127, %ki1_127, %ki2_127, %pad1_127, %out_v, %outidx_127) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 128: dilation=9 padding=36 kernel=(1,5,7) padding1=1
        %fidx_128 = arith.constant 128 : index
        %bidx_128 = arith.constant 379 : index
        %smidx_128 = arith.constant 799 : index
        %ssidx_128 = arith.constant 1219 : index
        %bias_128 = memref.load %in_v[%bidx_128] : memref<1511xi32>
        %smean_128 = memref.load %in_v[%smidx_128] : memref<1511xi32>
        %sscale_128 = memref.load %in_v[%ssidx_128] : memref<1511xi32>
        %dilation_128 = arith.constant 9 : i32
        %padding_128 = arith.constant 36 : i32
        %ki0_128 = arith.constant 1 : i32
        %ki1_128 = arith.constant 5 : i32
        %ki2_128 = arith.constant 7 : i32
        %pad1_128 = arith.constant 1 : i32
        %outidx_128 = arith.constant 128 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_128, %smean_128, %sscale_128, %dilation_128, %padding_128, %ki0_128, %ki1_128, %ki2_128, %pad1_128, %out_v, %outidx_128) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 129: dilation=9 padding=36 kernel=(1,5,8) padding1=0
        %fidx_129 = arith.constant 129 : index
        %bidx_129 = arith.constant 380 : index
        %smidx_129 = arith.constant 800 : index
        %ssidx_129 = arith.constant 1220 : index
        %bias_129 = memref.load %in_v[%bidx_129] : memref<1511xi32>
        %smean_129 = memref.load %in_v[%smidx_129] : memref<1511xi32>
        %sscale_129 = memref.load %in_v[%ssidx_129] : memref<1511xi32>
        %dilation_129 = arith.constant 9 : i32
        %padding_129 = arith.constant 36 : i32
        %ki0_129 = arith.constant 1 : i32
        %ki1_129 = arith.constant 5 : i32
        %ki2_129 = arith.constant 8 : i32
        %pad1_129 = arith.constant 0 : i32
        %outidx_129 = arith.constant 129 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_129, %smean_129, %sscale_129, %dilation_129, %padding_129, %ki0_129, %ki1_129, %ki2_129, %pad1_129, %out_v, %outidx_129) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 130: dilation=9 padding=36 kernel=(1,6,7) padding1=1
        %fidx_130 = arith.constant 130 : index
        %bidx_130 = arith.constant 381 : index
        %smidx_130 = arith.constant 801 : index
        %ssidx_130 = arith.constant 1221 : index
        %bias_130 = memref.load %in_v[%bidx_130] : memref<1511xi32>
        %smean_130 = memref.load %in_v[%smidx_130] : memref<1511xi32>
        %sscale_130 = memref.load %in_v[%ssidx_130] : memref<1511xi32>
        %dilation_130 = arith.constant 9 : i32
        %padding_130 = arith.constant 36 : i32
        %ki0_130 = arith.constant 1 : i32
        %ki1_130 = arith.constant 6 : i32
        %ki2_130 = arith.constant 7 : i32
        %pad1_130 = arith.constant 1 : i32
        %outidx_130 = arith.constant 130 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_130, %smean_130, %sscale_130, %dilation_130, %padding_130, %ki0_130, %ki1_130, %ki2_130, %pad1_130, %out_v, %outidx_130) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 131: dilation=9 padding=36 kernel=(1,6,8) padding1=0
        %fidx_131 = arith.constant 131 : index
        %bidx_131 = arith.constant 382 : index
        %smidx_131 = arith.constant 802 : index
        %ssidx_131 = arith.constant 1222 : index
        %bias_131 = memref.load %in_v[%bidx_131] : memref<1511xi32>
        %smean_131 = memref.load %in_v[%smidx_131] : memref<1511xi32>
        %sscale_131 = memref.load %in_v[%ssidx_131] : memref<1511xi32>
        %dilation_131 = arith.constant 9 : i32
        %padding_131 = arith.constant 36 : i32
        %ki0_131 = arith.constant 1 : i32
        %ki1_131 = arith.constant 6 : i32
        %ki2_131 = arith.constant 8 : i32
        %pad1_131 = arith.constant 0 : i32
        %outidx_131 = arith.constant 131 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_131, %smean_131, %sscale_131, %dilation_131, %padding_131, %ki0_131, %ki1_131, %ki2_131, %pad1_131, %out_v, %outidx_131) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 132: dilation=9 padding=36 kernel=(1,7,8) padding1=1
        %fidx_132 = arith.constant 132 : index
        %bidx_132 = arith.constant 383 : index
        %smidx_132 = arith.constant 803 : index
        %ssidx_132 = arith.constant 1223 : index
        %bias_132 = memref.load %in_v[%bidx_132] : memref<1511xi32>
        %smean_132 = memref.load %in_v[%smidx_132] : memref<1511xi32>
        %sscale_132 = memref.load %in_v[%ssidx_132] : memref<1511xi32>
        %dilation_132 = arith.constant 9 : i32
        %padding_132 = arith.constant 36 : i32
        %ki0_132 = arith.constant 1 : i32
        %ki1_132 = arith.constant 7 : i32
        %ki2_132 = arith.constant 8 : i32
        %pad1_132 = arith.constant 1 : i32
        %outidx_132 = arith.constant 132 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_132, %smean_132, %sscale_132, %dilation_132, %padding_132, %ki0_132, %ki1_132, %ki2_132, %pad1_132, %out_v, %outidx_132) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 133: dilation=9 padding=36 kernel=(2,3,4) padding1=0
        %fidx_133 = arith.constant 133 : index
        %bidx_133 = arith.constant 384 : index
        %smidx_133 = arith.constant 804 : index
        %ssidx_133 = arith.constant 1224 : index
        %bias_133 = memref.load %in_v[%bidx_133] : memref<1511xi32>
        %smean_133 = memref.load %in_v[%smidx_133] : memref<1511xi32>
        %sscale_133 = memref.load %in_v[%ssidx_133] : memref<1511xi32>
        %dilation_133 = arith.constant 9 : i32
        %padding_133 = arith.constant 36 : i32
        %ki0_133 = arith.constant 2 : i32
        %ki1_133 = arith.constant 3 : i32
        %ki2_133 = arith.constant 4 : i32
        %pad1_133 = arith.constant 0 : i32
        %outidx_133 = arith.constant 133 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_133, %smean_133, %sscale_133, %dilation_133, %padding_133, %ki0_133, %ki1_133, %ki2_133, %pad1_133, %out_v, %outidx_133) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 134: dilation=9 padding=36 kernel=(2,3,5) padding1=1
        %fidx_134 = arith.constant 134 : index
        %bidx_134 = arith.constant 385 : index
        %smidx_134 = arith.constant 805 : index
        %ssidx_134 = arith.constant 1225 : index
        %bias_134 = memref.load %in_v[%bidx_134] : memref<1511xi32>
        %smean_134 = memref.load %in_v[%smidx_134] : memref<1511xi32>
        %sscale_134 = memref.load %in_v[%ssidx_134] : memref<1511xi32>
        %dilation_134 = arith.constant 9 : i32
        %padding_134 = arith.constant 36 : i32
        %ki0_134 = arith.constant 2 : i32
        %ki1_134 = arith.constant 3 : i32
        %ki2_134 = arith.constant 5 : i32
        %pad1_134 = arith.constant 1 : i32
        %outidx_134 = arith.constant 134 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_134, %smean_134, %sscale_134, %dilation_134, %padding_134, %ki0_134, %ki1_134, %ki2_134, %pad1_134, %out_v, %outidx_134) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 135: dilation=9 padding=36 kernel=(2,3,6) padding1=0
        %fidx_135 = arith.constant 135 : index
        %bidx_135 = arith.constant 386 : index
        %smidx_135 = arith.constant 806 : index
        %ssidx_135 = arith.constant 1226 : index
        %bias_135 = memref.load %in_v[%bidx_135] : memref<1511xi32>
        %smean_135 = memref.load %in_v[%smidx_135] : memref<1511xi32>
        %sscale_135 = memref.load %in_v[%ssidx_135] : memref<1511xi32>
        %dilation_135 = arith.constant 9 : i32
        %padding_135 = arith.constant 36 : i32
        %ki0_135 = arith.constant 2 : i32
        %ki1_135 = arith.constant 3 : i32
        %ki2_135 = arith.constant 6 : i32
        %pad1_135 = arith.constant 0 : i32
        %outidx_135 = arith.constant 135 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_135, %smean_135, %sscale_135, %dilation_135, %padding_135, %ki0_135, %ki1_135, %ki2_135, %pad1_135, %out_v, %outidx_135) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 136: dilation=9 padding=36 kernel=(2,3,7) padding1=1
        %fidx_136 = arith.constant 136 : index
        %bidx_136 = arith.constant 387 : index
        %smidx_136 = arith.constant 807 : index
        %ssidx_136 = arith.constant 1227 : index
        %bias_136 = memref.load %in_v[%bidx_136] : memref<1511xi32>
        %smean_136 = memref.load %in_v[%smidx_136] : memref<1511xi32>
        %sscale_136 = memref.load %in_v[%ssidx_136] : memref<1511xi32>
        %dilation_136 = arith.constant 9 : i32
        %padding_136 = arith.constant 36 : i32
        %ki0_136 = arith.constant 2 : i32
        %ki1_136 = arith.constant 3 : i32
        %ki2_136 = arith.constant 7 : i32
        %pad1_136 = arith.constant 1 : i32
        %outidx_136 = arith.constant 136 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_136, %smean_136, %sscale_136, %dilation_136, %padding_136, %ki0_136, %ki1_136, %ki2_136, %pad1_136, %out_v, %outidx_136) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 137: dilation=9 padding=36 kernel=(2,3,8) padding1=0
        %fidx_137 = arith.constant 137 : index
        %bidx_137 = arith.constant 388 : index
        %smidx_137 = arith.constant 808 : index
        %ssidx_137 = arith.constant 1228 : index
        %bias_137 = memref.load %in_v[%bidx_137] : memref<1511xi32>
        %smean_137 = memref.load %in_v[%smidx_137] : memref<1511xi32>
        %sscale_137 = memref.load %in_v[%ssidx_137] : memref<1511xi32>
        %dilation_137 = arith.constant 9 : i32
        %padding_137 = arith.constant 36 : i32
        %ki0_137 = arith.constant 2 : i32
        %ki1_137 = arith.constant 3 : i32
        %ki2_137 = arith.constant 8 : i32
        %pad1_137 = arith.constant 0 : i32
        %outidx_137 = arith.constant 137 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_137, %smean_137, %sscale_137, %dilation_137, %padding_137, %ki0_137, %ki1_137, %ki2_137, %pad1_137, %out_v, %outidx_137) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 138: dilation=9 padding=36 kernel=(2,4,5) padding1=1
        %fidx_138 = arith.constant 138 : index
        %bidx_138 = arith.constant 389 : index
        %smidx_138 = arith.constant 809 : index
        %ssidx_138 = arith.constant 1229 : index
        %bias_138 = memref.load %in_v[%bidx_138] : memref<1511xi32>
        %smean_138 = memref.load %in_v[%smidx_138] : memref<1511xi32>
        %sscale_138 = memref.load %in_v[%ssidx_138] : memref<1511xi32>
        %dilation_138 = arith.constant 9 : i32
        %padding_138 = arith.constant 36 : i32
        %ki0_138 = arith.constant 2 : i32
        %ki1_138 = arith.constant 4 : i32
        %ki2_138 = arith.constant 5 : i32
        %pad1_138 = arith.constant 1 : i32
        %outidx_138 = arith.constant 138 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_138, %smean_138, %sscale_138, %dilation_138, %padding_138, %ki0_138, %ki1_138, %ki2_138, %pad1_138, %out_v, %outidx_138) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 139: dilation=9 padding=36 kernel=(2,4,6) padding1=0
        %fidx_139 = arith.constant 139 : index
        %bidx_139 = arith.constant 390 : index
        %smidx_139 = arith.constant 810 : index
        %ssidx_139 = arith.constant 1230 : index
        %bias_139 = memref.load %in_v[%bidx_139] : memref<1511xi32>
        %smean_139 = memref.load %in_v[%smidx_139] : memref<1511xi32>
        %sscale_139 = memref.load %in_v[%ssidx_139] : memref<1511xi32>
        %dilation_139 = arith.constant 9 : i32
        %padding_139 = arith.constant 36 : i32
        %ki0_139 = arith.constant 2 : i32
        %ki1_139 = arith.constant 4 : i32
        %ki2_139 = arith.constant 6 : i32
        %pad1_139 = arith.constant 0 : i32
        %outidx_139 = arith.constant 139 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_139, %smean_139, %sscale_139, %dilation_139, %padding_139, %ki0_139, %ki1_139, %ki2_139, %pad1_139, %out_v, %outidx_139) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 140: dilation=9 padding=36 kernel=(2,4,7) padding1=1
        %fidx_140 = arith.constant 140 : index
        %bidx_140 = arith.constant 391 : index
        %smidx_140 = arith.constant 811 : index
        %ssidx_140 = arith.constant 1231 : index
        %bias_140 = memref.load %in_v[%bidx_140] : memref<1511xi32>
        %smean_140 = memref.load %in_v[%smidx_140] : memref<1511xi32>
        %sscale_140 = memref.load %in_v[%ssidx_140] : memref<1511xi32>
        %dilation_140 = arith.constant 9 : i32
        %padding_140 = arith.constant 36 : i32
        %ki0_140 = arith.constant 2 : i32
        %ki1_140 = arith.constant 4 : i32
        %ki2_140 = arith.constant 7 : i32
        %pad1_140 = arith.constant 1 : i32
        %outidx_140 = arith.constant 140 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_140, %smean_140, %sscale_140, %dilation_140, %padding_140, %ki0_140, %ki1_140, %ki2_140, %pad1_140, %out_v, %outidx_140) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 141: dilation=9 padding=36 kernel=(2,4,8) padding1=0
        %fidx_141 = arith.constant 141 : index
        %bidx_141 = arith.constant 392 : index
        %smidx_141 = arith.constant 812 : index
        %ssidx_141 = arith.constant 1232 : index
        %bias_141 = memref.load %in_v[%bidx_141] : memref<1511xi32>
        %smean_141 = memref.load %in_v[%smidx_141] : memref<1511xi32>
        %sscale_141 = memref.load %in_v[%ssidx_141] : memref<1511xi32>
        %dilation_141 = arith.constant 9 : i32
        %padding_141 = arith.constant 36 : i32
        %ki0_141 = arith.constant 2 : i32
        %ki1_141 = arith.constant 4 : i32
        %ki2_141 = arith.constant 8 : i32
        %pad1_141 = arith.constant 0 : i32
        %outidx_141 = arith.constant 141 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_141, %smean_141, %sscale_141, %dilation_141, %padding_141, %ki0_141, %ki1_141, %ki2_141, %pad1_141, %out_v, %outidx_141) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 142: dilation=9 padding=36 kernel=(2,5,6) padding1=1
        %fidx_142 = arith.constant 142 : index
        %bidx_142 = arith.constant 393 : index
        %smidx_142 = arith.constant 813 : index
        %ssidx_142 = arith.constant 1233 : index
        %bias_142 = memref.load %in_v[%bidx_142] : memref<1511xi32>
        %smean_142 = memref.load %in_v[%smidx_142] : memref<1511xi32>
        %sscale_142 = memref.load %in_v[%ssidx_142] : memref<1511xi32>
        %dilation_142 = arith.constant 9 : i32
        %padding_142 = arith.constant 36 : i32
        %ki0_142 = arith.constant 2 : i32
        %ki1_142 = arith.constant 5 : i32
        %ki2_142 = arith.constant 6 : i32
        %pad1_142 = arith.constant 1 : i32
        %outidx_142 = arith.constant 142 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_142, %smean_142, %sscale_142, %dilation_142, %padding_142, %ki0_142, %ki1_142, %ki2_142, %pad1_142, %out_v, %outidx_142) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 143: dilation=9 padding=36 kernel=(2,5,7) padding1=0
        %fidx_143 = arith.constant 143 : index
        %bidx_143 = arith.constant 394 : index
        %smidx_143 = arith.constant 814 : index
        %ssidx_143 = arith.constant 1234 : index
        %bias_143 = memref.load %in_v[%bidx_143] : memref<1511xi32>
        %smean_143 = memref.load %in_v[%smidx_143] : memref<1511xi32>
        %sscale_143 = memref.load %in_v[%ssidx_143] : memref<1511xi32>
        %dilation_143 = arith.constant 9 : i32
        %padding_143 = arith.constant 36 : i32
        %ki0_143 = arith.constant 2 : i32
        %ki1_143 = arith.constant 5 : i32
        %ki2_143 = arith.constant 7 : i32
        %pad1_143 = arith.constant 0 : i32
        %outidx_143 = arith.constant 143 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_143, %smean_143, %sscale_143, %dilation_143, %padding_143, %ki0_143, %ki1_143, %ki2_143, %pad1_143, %out_v, %outidx_143) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 144: dilation=9 padding=36 kernel=(2,5,8) padding1=1
        %fidx_144 = arith.constant 144 : index
        %bidx_144 = arith.constant 395 : index
        %smidx_144 = arith.constant 815 : index
        %ssidx_144 = arith.constant 1235 : index
        %bias_144 = memref.load %in_v[%bidx_144] : memref<1511xi32>
        %smean_144 = memref.load %in_v[%smidx_144] : memref<1511xi32>
        %sscale_144 = memref.load %in_v[%ssidx_144] : memref<1511xi32>
        %dilation_144 = arith.constant 9 : i32
        %padding_144 = arith.constant 36 : i32
        %ki0_144 = arith.constant 2 : i32
        %ki1_144 = arith.constant 5 : i32
        %ki2_144 = arith.constant 8 : i32
        %pad1_144 = arith.constant 1 : i32
        %outidx_144 = arith.constant 144 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_144, %smean_144, %sscale_144, %dilation_144, %padding_144, %ki0_144, %ki1_144, %ki2_144, %pad1_144, %out_v, %outidx_144) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 145: dilation=9 padding=36 kernel=(2,6,7) padding1=0
        %fidx_145 = arith.constant 145 : index
        %bidx_145 = arith.constant 396 : index
        %smidx_145 = arith.constant 816 : index
        %ssidx_145 = arith.constant 1236 : index
        %bias_145 = memref.load %in_v[%bidx_145] : memref<1511xi32>
        %smean_145 = memref.load %in_v[%smidx_145] : memref<1511xi32>
        %sscale_145 = memref.load %in_v[%ssidx_145] : memref<1511xi32>
        %dilation_145 = arith.constant 9 : i32
        %padding_145 = arith.constant 36 : i32
        %ki0_145 = arith.constant 2 : i32
        %ki1_145 = arith.constant 6 : i32
        %ki2_145 = arith.constant 7 : i32
        %pad1_145 = arith.constant 0 : i32
        %outidx_145 = arith.constant 145 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_145, %smean_145, %sscale_145, %dilation_145, %padding_145, %ki0_145, %ki1_145, %ki2_145, %pad1_145, %out_v, %outidx_145) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 146: dilation=9 padding=36 kernel=(2,6,8) padding1=1
        %fidx_146 = arith.constant 146 : index
        %bidx_146 = arith.constant 397 : index
        %smidx_146 = arith.constant 817 : index
        %ssidx_146 = arith.constant 1237 : index
        %bias_146 = memref.load %in_v[%bidx_146] : memref<1511xi32>
        %smean_146 = memref.load %in_v[%smidx_146] : memref<1511xi32>
        %sscale_146 = memref.load %in_v[%ssidx_146] : memref<1511xi32>
        %dilation_146 = arith.constant 9 : i32
        %padding_146 = arith.constant 36 : i32
        %ki0_146 = arith.constant 2 : i32
        %ki1_146 = arith.constant 6 : i32
        %ki2_146 = arith.constant 8 : i32
        %pad1_146 = arith.constant 1 : i32
        %outidx_146 = arith.constant 146 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_146, %smean_146, %sscale_146, %dilation_146, %padding_146, %ki0_146, %ki1_146, %ki2_146, %pad1_146, %out_v, %outidx_146) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 147: dilation=9 padding=36 kernel=(2,7,8) padding1=0
        %fidx_147 = arith.constant 147 : index
        %bidx_147 = arith.constant 398 : index
        %smidx_147 = arith.constant 818 : index
        %ssidx_147 = arith.constant 1238 : index
        %bias_147 = memref.load %in_v[%bidx_147] : memref<1511xi32>
        %smean_147 = memref.load %in_v[%smidx_147] : memref<1511xi32>
        %sscale_147 = memref.load %in_v[%ssidx_147] : memref<1511xi32>
        %dilation_147 = arith.constant 9 : i32
        %padding_147 = arith.constant 36 : i32
        %ki0_147 = arith.constant 2 : i32
        %ki1_147 = arith.constant 7 : i32
        %ki2_147 = arith.constant 8 : i32
        %pad1_147 = arith.constant 0 : i32
        %outidx_147 = arith.constant 147 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_147, %smean_147, %sscale_147, %dilation_147, %padding_147, %ki0_147, %ki1_147, %ki2_147, %pad1_147, %out_v, %outidx_147) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 148: dilation=9 padding=36 kernel=(3,4,5) padding1=1
        %fidx_148 = arith.constant 148 : index
        %bidx_148 = arith.constant 399 : index
        %smidx_148 = arith.constant 819 : index
        %ssidx_148 = arith.constant 1239 : index
        %bias_148 = memref.load %in_v[%bidx_148] : memref<1511xi32>
        %smean_148 = memref.load %in_v[%smidx_148] : memref<1511xi32>
        %sscale_148 = memref.load %in_v[%ssidx_148] : memref<1511xi32>
        %dilation_148 = arith.constant 9 : i32
        %padding_148 = arith.constant 36 : i32
        %ki0_148 = arith.constant 3 : i32
        %ki1_148 = arith.constant 4 : i32
        %ki2_148 = arith.constant 5 : i32
        %pad1_148 = arith.constant 1 : i32
        %outidx_148 = arith.constant 148 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_148, %smean_148, %sscale_148, %dilation_148, %padding_148, %ki0_148, %ki1_148, %ki2_148, %pad1_148, %out_v, %outidx_148) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 149: dilation=9 padding=36 kernel=(3,4,6) padding1=0
        %fidx_149 = arith.constant 149 : index
        %bidx_149 = arith.constant 400 : index
        %smidx_149 = arith.constant 820 : index
        %ssidx_149 = arith.constant 1240 : index
        %bias_149 = memref.load %in_v[%bidx_149] : memref<1511xi32>
        %smean_149 = memref.load %in_v[%smidx_149] : memref<1511xi32>
        %sscale_149 = memref.load %in_v[%ssidx_149] : memref<1511xi32>
        %dilation_149 = arith.constant 9 : i32
        %padding_149 = arith.constant 36 : i32
        %ki0_149 = arith.constant 3 : i32
        %ki1_149 = arith.constant 4 : i32
        %ki2_149 = arith.constant 6 : i32
        %pad1_149 = arith.constant 0 : i32
        %outidx_149 = arith.constant 149 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_149, %smean_149, %sscale_149, %dilation_149, %padding_149, %ki0_149, %ki1_149, %ki2_149, %pad1_149, %out_v, %outidx_149) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 150: dilation=9 padding=36 kernel=(3,4,7) padding1=1
        %fidx_150 = arith.constant 150 : index
        %bidx_150 = arith.constant 401 : index
        %smidx_150 = arith.constant 821 : index
        %ssidx_150 = arith.constant 1241 : index
        %bias_150 = memref.load %in_v[%bidx_150] : memref<1511xi32>
        %smean_150 = memref.load %in_v[%smidx_150] : memref<1511xi32>
        %sscale_150 = memref.load %in_v[%ssidx_150] : memref<1511xi32>
        %dilation_150 = arith.constant 9 : i32
        %padding_150 = arith.constant 36 : i32
        %ki0_150 = arith.constant 3 : i32
        %ki1_150 = arith.constant 4 : i32
        %ki2_150 = arith.constant 7 : i32
        %pad1_150 = arith.constant 1 : i32
        %outidx_150 = arith.constant 150 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_150, %smean_150, %sscale_150, %dilation_150, %padding_150, %ki0_150, %ki1_150, %ki2_150, %pad1_150, %out_v, %outidx_150) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 151: dilation=9 padding=36 kernel=(3,4,8) padding1=0
        %fidx_151 = arith.constant 151 : index
        %bidx_151 = arith.constant 402 : index
        %smidx_151 = arith.constant 822 : index
        %ssidx_151 = arith.constant 1242 : index
        %bias_151 = memref.load %in_v[%bidx_151] : memref<1511xi32>
        %smean_151 = memref.load %in_v[%smidx_151] : memref<1511xi32>
        %sscale_151 = memref.load %in_v[%ssidx_151] : memref<1511xi32>
        %dilation_151 = arith.constant 9 : i32
        %padding_151 = arith.constant 36 : i32
        %ki0_151 = arith.constant 3 : i32
        %ki1_151 = arith.constant 4 : i32
        %ki2_151 = arith.constant 8 : i32
        %pad1_151 = arith.constant 0 : i32
        %outidx_151 = arith.constant 151 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_151, %smean_151, %sscale_151, %dilation_151, %padding_151, %ki0_151, %ki1_151, %ki2_151, %pad1_151, %out_v, %outidx_151) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 152: dilation=9 padding=36 kernel=(3,5,6) padding1=1
        %fidx_152 = arith.constant 152 : index
        %bidx_152 = arith.constant 403 : index
        %smidx_152 = arith.constant 823 : index
        %ssidx_152 = arith.constant 1243 : index
        %bias_152 = memref.load %in_v[%bidx_152] : memref<1511xi32>
        %smean_152 = memref.load %in_v[%smidx_152] : memref<1511xi32>
        %sscale_152 = memref.load %in_v[%ssidx_152] : memref<1511xi32>
        %dilation_152 = arith.constant 9 : i32
        %padding_152 = arith.constant 36 : i32
        %ki0_152 = arith.constant 3 : i32
        %ki1_152 = arith.constant 5 : i32
        %ki2_152 = arith.constant 6 : i32
        %pad1_152 = arith.constant 1 : i32
        %outidx_152 = arith.constant 152 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_152, %smean_152, %sscale_152, %dilation_152, %padding_152, %ki0_152, %ki1_152, %ki2_152, %pad1_152, %out_v, %outidx_152) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 153: dilation=9 padding=36 kernel=(3,5,7) padding1=0
        %fidx_153 = arith.constant 153 : index
        %bidx_153 = arith.constant 404 : index
        %smidx_153 = arith.constant 824 : index
        %ssidx_153 = arith.constant 1244 : index
        %bias_153 = memref.load %in_v[%bidx_153] : memref<1511xi32>
        %smean_153 = memref.load %in_v[%smidx_153] : memref<1511xi32>
        %sscale_153 = memref.load %in_v[%ssidx_153] : memref<1511xi32>
        %dilation_153 = arith.constant 9 : i32
        %padding_153 = arith.constant 36 : i32
        %ki0_153 = arith.constant 3 : i32
        %ki1_153 = arith.constant 5 : i32
        %ki2_153 = arith.constant 7 : i32
        %pad1_153 = arith.constant 0 : i32
        %outidx_153 = arith.constant 153 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_153, %smean_153, %sscale_153, %dilation_153, %padding_153, %ki0_153, %ki1_153, %ki2_153, %pad1_153, %out_v, %outidx_153) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 154: dilation=9 padding=36 kernel=(3,5,8) padding1=1
        %fidx_154 = arith.constant 154 : index
        %bidx_154 = arith.constant 405 : index
        %smidx_154 = arith.constant 825 : index
        %ssidx_154 = arith.constant 1245 : index
        %bias_154 = memref.load %in_v[%bidx_154] : memref<1511xi32>
        %smean_154 = memref.load %in_v[%smidx_154] : memref<1511xi32>
        %sscale_154 = memref.load %in_v[%ssidx_154] : memref<1511xi32>
        %dilation_154 = arith.constant 9 : i32
        %padding_154 = arith.constant 36 : i32
        %ki0_154 = arith.constant 3 : i32
        %ki1_154 = arith.constant 5 : i32
        %ki2_154 = arith.constant 8 : i32
        %pad1_154 = arith.constant 1 : i32
        %outidx_154 = arith.constant 154 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_154, %smean_154, %sscale_154, %dilation_154, %padding_154, %ki0_154, %ki1_154, %ki2_154, %pad1_154, %out_v, %outidx_154) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 155: dilation=9 padding=36 kernel=(3,6,7) padding1=0
        %fidx_155 = arith.constant 155 : index
        %bidx_155 = arith.constant 406 : index
        %smidx_155 = arith.constant 826 : index
        %ssidx_155 = arith.constant 1246 : index
        %bias_155 = memref.load %in_v[%bidx_155] : memref<1511xi32>
        %smean_155 = memref.load %in_v[%smidx_155] : memref<1511xi32>
        %sscale_155 = memref.load %in_v[%ssidx_155] : memref<1511xi32>
        %dilation_155 = arith.constant 9 : i32
        %padding_155 = arith.constant 36 : i32
        %ki0_155 = arith.constant 3 : i32
        %ki1_155 = arith.constant 6 : i32
        %ki2_155 = arith.constant 7 : i32
        %pad1_155 = arith.constant 0 : i32
        %outidx_155 = arith.constant 155 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_155, %smean_155, %sscale_155, %dilation_155, %padding_155, %ki0_155, %ki1_155, %ki2_155, %pad1_155, %out_v, %outidx_155) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 156: dilation=9 padding=36 kernel=(3,6,8) padding1=1
        %fidx_156 = arith.constant 156 : index
        %bidx_156 = arith.constant 407 : index
        %smidx_156 = arith.constant 827 : index
        %ssidx_156 = arith.constant 1247 : index
        %bias_156 = memref.load %in_v[%bidx_156] : memref<1511xi32>
        %smean_156 = memref.load %in_v[%smidx_156] : memref<1511xi32>
        %sscale_156 = memref.load %in_v[%ssidx_156] : memref<1511xi32>
        %dilation_156 = arith.constant 9 : i32
        %padding_156 = arith.constant 36 : i32
        %ki0_156 = arith.constant 3 : i32
        %ki1_156 = arith.constant 6 : i32
        %ki2_156 = arith.constant 8 : i32
        %pad1_156 = arith.constant 1 : i32
        %outidx_156 = arith.constant 156 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_156, %smean_156, %sscale_156, %dilation_156, %padding_156, %ki0_156, %ki1_156, %ki2_156, %pad1_156, %out_v, %outidx_156) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 157: dilation=9 padding=36 kernel=(3,7,8) padding1=0
        %fidx_157 = arith.constant 157 : index
        %bidx_157 = arith.constant 408 : index
        %smidx_157 = arith.constant 828 : index
        %ssidx_157 = arith.constant 1248 : index
        %bias_157 = memref.load %in_v[%bidx_157] : memref<1511xi32>
        %smean_157 = memref.load %in_v[%smidx_157] : memref<1511xi32>
        %sscale_157 = memref.load %in_v[%ssidx_157] : memref<1511xi32>
        %dilation_157 = arith.constant 9 : i32
        %padding_157 = arith.constant 36 : i32
        %ki0_157 = arith.constant 3 : i32
        %ki1_157 = arith.constant 7 : i32
        %ki2_157 = arith.constant 8 : i32
        %pad1_157 = arith.constant 0 : i32
        %outidx_157 = arith.constant 157 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_157, %smean_157, %sscale_157, %dilation_157, %padding_157, %ki0_157, %ki1_157, %ki2_157, %pad1_157, %out_v, %outidx_157) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 158: dilation=9 padding=36 kernel=(4,5,6) padding1=1
        %fidx_158 = arith.constant 158 : index
        %bidx_158 = arith.constant 409 : index
        %smidx_158 = arith.constant 829 : index
        %ssidx_158 = arith.constant 1249 : index
        %bias_158 = memref.load %in_v[%bidx_158] : memref<1511xi32>
        %smean_158 = memref.load %in_v[%smidx_158] : memref<1511xi32>
        %sscale_158 = memref.load %in_v[%ssidx_158] : memref<1511xi32>
        %dilation_158 = arith.constant 9 : i32
        %padding_158 = arith.constant 36 : i32
        %ki0_158 = arith.constant 4 : i32
        %ki1_158 = arith.constant 5 : i32
        %ki2_158 = arith.constant 6 : i32
        %pad1_158 = arith.constant 1 : i32
        %outidx_158 = arith.constant 158 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_158, %smean_158, %sscale_158, %dilation_158, %padding_158, %ki0_158, %ki1_158, %ki2_158, %pad1_158, %out_v, %outidx_158) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 159: dilation=9 padding=36 kernel=(4,5,7) padding1=0
        %fidx_159 = arith.constant 159 : index
        %bidx_159 = arith.constant 410 : index
        %smidx_159 = arith.constant 830 : index
        %ssidx_159 = arith.constant 1250 : index
        %bias_159 = memref.load %in_v[%bidx_159] : memref<1511xi32>
        %smean_159 = memref.load %in_v[%smidx_159] : memref<1511xi32>
        %sscale_159 = memref.load %in_v[%ssidx_159] : memref<1511xi32>
        %dilation_159 = arith.constant 9 : i32
        %padding_159 = arith.constant 36 : i32
        %ki0_159 = arith.constant 4 : i32
        %ki1_159 = arith.constant 5 : i32
        %ki2_159 = arith.constant 7 : i32
        %pad1_159 = arith.constant 0 : i32
        %outidx_159 = arith.constant 159 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_159, %smean_159, %sscale_159, %dilation_159, %padding_159, %ki0_159, %ki1_159, %ki2_159, %pad1_159, %out_v, %outidx_159) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 160: dilation=9 padding=36 kernel=(4,5,8) padding1=1
        %fidx_160 = arith.constant 160 : index
        %bidx_160 = arith.constant 411 : index
        %smidx_160 = arith.constant 831 : index
        %ssidx_160 = arith.constant 1251 : index
        %bias_160 = memref.load %in_v[%bidx_160] : memref<1511xi32>
        %smean_160 = memref.load %in_v[%smidx_160] : memref<1511xi32>
        %sscale_160 = memref.load %in_v[%ssidx_160] : memref<1511xi32>
        %dilation_160 = arith.constant 9 : i32
        %padding_160 = arith.constant 36 : i32
        %ki0_160 = arith.constant 4 : i32
        %ki1_160 = arith.constant 5 : i32
        %ki2_160 = arith.constant 8 : i32
        %pad1_160 = arith.constant 1 : i32
        %outidx_160 = arith.constant 160 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_160, %smean_160, %sscale_160, %dilation_160, %padding_160, %ki0_160, %ki1_160, %ki2_160, %pad1_160, %out_v, %outidx_160) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 161: dilation=9 padding=36 kernel=(4,6,7) padding1=0
        %fidx_161 = arith.constant 161 : index
        %bidx_161 = arith.constant 412 : index
        %smidx_161 = arith.constant 832 : index
        %ssidx_161 = arith.constant 1252 : index
        %bias_161 = memref.load %in_v[%bidx_161] : memref<1511xi32>
        %smean_161 = memref.load %in_v[%smidx_161] : memref<1511xi32>
        %sscale_161 = memref.load %in_v[%ssidx_161] : memref<1511xi32>
        %dilation_161 = arith.constant 9 : i32
        %padding_161 = arith.constant 36 : i32
        %ki0_161 = arith.constant 4 : i32
        %ki1_161 = arith.constant 6 : i32
        %ki2_161 = arith.constant 7 : i32
        %pad1_161 = arith.constant 0 : i32
        %outidx_161 = arith.constant 161 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_161, %smean_161, %sscale_161, %dilation_161, %padding_161, %ki0_161, %ki1_161, %ki2_161, %pad1_161, %out_v, %outidx_161) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 162: dilation=9 padding=36 kernel=(4,6,8) padding1=1
        %fidx_162 = arith.constant 162 : index
        %bidx_162 = arith.constant 413 : index
        %smidx_162 = arith.constant 833 : index
        %ssidx_162 = arith.constant 1253 : index
        %bias_162 = memref.load %in_v[%bidx_162] : memref<1511xi32>
        %smean_162 = memref.load %in_v[%smidx_162] : memref<1511xi32>
        %sscale_162 = memref.load %in_v[%ssidx_162] : memref<1511xi32>
        %dilation_162 = arith.constant 9 : i32
        %padding_162 = arith.constant 36 : i32
        %ki0_162 = arith.constant 4 : i32
        %ki1_162 = arith.constant 6 : i32
        %ki2_162 = arith.constant 8 : i32
        %pad1_162 = arith.constant 1 : i32
        %outidx_162 = arith.constant 162 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_162, %smean_162, %sscale_162, %dilation_162, %padding_162, %ki0_162, %ki1_162, %ki2_162, %pad1_162, %out_v, %outidx_162) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 163: dilation=9 padding=36 kernel=(4,7,8) padding1=0
        %fidx_163 = arith.constant 163 : index
        %bidx_163 = arith.constant 414 : index
        %smidx_163 = arith.constant 834 : index
        %ssidx_163 = arith.constant 1254 : index
        %bias_163 = memref.load %in_v[%bidx_163] : memref<1511xi32>
        %smean_163 = memref.load %in_v[%smidx_163] : memref<1511xi32>
        %sscale_163 = memref.load %in_v[%ssidx_163] : memref<1511xi32>
        %dilation_163 = arith.constant 9 : i32
        %padding_163 = arith.constant 36 : i32
        %ki0_163 = arith.constant 4 : i32
        %ki1_163 = arith.constant 7 : i32
        %ki2_163 = arith.constant 8 : i32
        %pad1_163 = arith.constant 0 : i32
        %outidx_163 = arith.constant 163 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_163, %smean_163, %sscale_163, %dilation_163, %padding_163, %ki0_163, %ki1_163, %ki2_163, %pad1_163, %out_v, %outidx_163) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 164: dilation=9 padding=36 kernel=(5,6,7) padding1=1
        %fidx_164 = arith.constant 164 : index
        %bidx_164 = arith.constant 415 : index
        %smidx_164 = arith.constant 835 : index
        %ssidx_164 = arith.constant 1255 : index
        %bias_164 = memref.load %in_v[%bidx_164] : memref<1511xi32>
        %smean_164 = memref.load %in_v[%smidx_164] : memref<1511xi32>
        %sscale_164 = memref.load %in_v[%ssidx_164] : memref<1511xi32>
        %dilation_164 = arith.constant 9 : i32
        %padding_164 = arith.constant 36 : i32
        %ki0_164 = arith.constant 5 : i32
        %ki1_164 = arith.constant 6 : i32
        %ki2_164 = arith.constant 7 : i32
        %pad1_164 = arith.constant 1 : i32
        %outidx_164 = arith.constant 164 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_164, %smean_164, %sscale_164, %dilation_164, %padding_164, %ki0_164, %ki1_164, %ki2_164, %pad1_164, %out_v, %outidx_164) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 165: dilation=9 padding=36 kernel=(5,6,8) padding1=0
        %fidx_165 = arith.constant 165 : index
        %bidx_165 = arith.constant 416 : index
        %smidx_165 = arith.constant 836 : index
        %ssidx_165 = arith.constant 1256 : index
        %bias_165 = memref.load %in_v[%bidx_165] : memref<1511xi32>
        %smean_165 = memref.load %in_v[%smidx_165] : memref<1511xi32>
        %sscale_165 = memref.load %in_v[%ssidx_165] : memref<1511xi32>
        %dilation_165 = arith.constant 9 : i32
        %padding_165 = arith.constant 36 : i32
        %ki0_165 = arith.constant 5 : i32
        %ki1_165 = arith.constant 6 : i32
        %ki2_165 = arith.constant 8 : i32
        %pad1_165 = arith.constant 0 : i32
        %outidx_165 = arith.constant 165 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_165, %smean_165, %sscale_165, %dilation_165, %padding_165, %ki0_165, %ki1_165, %ki2_165, %pad1_165, %out_v, %outidx_165) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 166: dilation=9 padding=36 kernel=(5,7,8) padding1=1
        %fidx_166 = arith.constant 166 : index
        %bidx_166 = arith.constant 417 : index
        %smidx_166 = arith.constant 837 : index
        %ssidx_166 = arith.constant 1257 : index
        %bias_166 = memref.load %in_v[%bidx_166] : memref<1511xi32>
        %smean_166 = memref.load %in_v[%smidx_166] : memref<1511xi32>
        %sscale_166 = memref.load %in_v[%ssidx_166] : memref<1511xi32>
        %dilation_166 = arith.constant 9 : i32
        %padding_166 = arith.constant 36 : i32
        %ki0_166 = arith.constant 5 : i32
        %ki1_166 = arith.constant 7 : i32
        %ki2_166 = arith.constant 8 : i32
        %pad1_166 = arith.constant 1 : i32
        %outidx_166 = arith.constant 166 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_166, %smean_166, %sscale_166, %dilation_166, %padding_166, %ki0_166, %ki1_166, %ki2_166, %pad1_166, %out_v, %outidx_166) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 167: dilation=9 padding=36 kernel=(6,7,8) padding1=0
        %fidx_167 = arith.constant 167 : index
        %bidx_167 = arith.constant 418 : index
        %smidx_167 = arith.constant 838 : index
        %ssidx_167 = arith.constant 1258 : index
        %bias_167 = memref.load %in_v[%bidx_167] : memref<1511xi32>
        %smean_167 = memref.load %in_v[%smidx_167] : memref<1511xi32>
        %sscale_167 = memref.load %in_v[%ssidx_167] : memref<1511xi32>
        %dilation_167 = arith.constant 9 : i32
        %padding_167 = arith.constant 36 : i32
        %ki0_167 = arith.constant 6 : i32
        %ki1_167 = arith.constant 7 : i32
        %ki2_167 = arith.constant 8 : i32
        %pad1_167 = arith.constant 0 : i32
        %outidx_167 = arith.constant 167 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_167, %smean_167, %sscale_167, %dilation_167, %padding_167, %ki0_167, %ki1_167, %ki2_167, %pad1_167, %out_v, %outidx_167) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 168: dilation=14 padding=56 kernel=(0,1,2) padding1=0
        %fidx_168 = arith.constant 168 : index
        %bidx_168 = arith.constant 419 : index
        %smidx_168 = arith.constant 839 : index
        %ssidx_168 = arith.constant 1259 : index
        %bias_168 = memref.load %in_v[%bidx_168] : memref<1511xi32>
        %smean_168 = memref.load %in_v[%smidx_168] : memref<1511xi32>
        %sscale_168 = memref.load %in_v[%ssidx_168] : memref<1511xi32>
        %dilation_168 = arith.constant 14 : i32
        %padding_168 = arith.constant 56 : i32
        %ki0_168 = arith.constant 0 : i32
        %ki1_168 = arith.constant 1 : i32
        %ki2_168 = arith.constant 2 : i32
        %pad1_168 = arith.constant 0 : i32
        %outidx_168 = arith.constant 168 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_168, %smean_168, %sscale_168, %dilation_168, %padding_168, %ki0_168, %ki1_168, %ki2_168, %pad1_168, %out_v, %outidx_168) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 169: dilation=14 padding=56 kernel=(0,1,3) padding1=1
        %fidx_169 = arith.constant 169 : index
        %bidx_169 = arith.constant 420 : index
        %smidx_169 = arith.constant 840 : index
        %ssidx_169 = arith.constant 1260 : index
        %bias_169 = memref.load %in_v[%bidx_169] : memref<1511xi32>
        %smean_169 = memref.load %in_v[%smidx_169] : memref<1511xi32>
        %sscale_169 = memref.load %in_v[%ssidx_169] : memref<1511xi32>
        %dilation_169 = arith.constant 14 : i32
        %padding_169 = arith.constant 56 : i32
        %ki0_169 = arith.constant 0 : i32
        %ki1_169 = arith.constant 1 : i32
        %ki2_169 = arith.constant 3 : i32
        %pad1_169 = arith.constant 1 : i32
        %outidx_169 = arith.constant 169 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_169, %smean_169, %sscale_169, %dilation_169, %padding_169, %ki0_169, %ki1_169, %ki2_169, %pad1_169, %out_v, %outidx_169) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 170: dilation=14 padding=56 kernel=(0,1,4) padding1=0
        %fidx_170 = arith.constant 170 : index
        %bidx_170 = arith.constant 421 : index
        %smidx_170 = arith.constant 841 : index
        %ssidx_170 = arith.constant 1261 : index
        %bias_170 = memref.load %in_v[%bidx_170] : memref<1511xi32>
        %smean_170 = memref.load %in_v[%smidx_170] : memref<1511xi32>
        %sscale_170 = memref.load %in_v[%ssidx_170] : memref<1511xi32>
        %dilation_170 = arith.constant 14 : i32
        %padding_170 = arith.constant 56 : i32
        %ki0_170 = arith.constant 0 : i32
        %ki1_170 = arith.constant 1 : i32
        %ki2_170 = arith.constant 4 : i32
        %pad1_170 = arith.constant 0 : i32
        %outidx_170 = arith.constant 170 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_170, %smean_170, %sscale_170, %dilation_170, %padding_170, %ki0_170, %ki1_170, %ki2_170, %pad1_170, %out_v, %outidx_170) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 171: dilation=14 padding=56 kernel=(0,1,5) padding1=1
        %fidx_171 = arith.constant 171 : index
        %bidx_171 = arith.constant 422 : index
        %smidx_171 = arith.constant 842 : index
        %ssidx_171 = arith.constant 1262 : index
        %bias_171 = memref.load %in_v[%bidx_171] : memref<1511xi32>
        %smean_171 = memref.load %in_v[%smidx_171] : memref<1511xi32>
        %sscale_171 = memref.load %in_v[%ssidx_171] : memref<1511xi32>
        %dilation_171 = arith.constant 14 : i32
        %padding_171 = arith.constant 56 : i32
        %ki0_171 = arith.constant 0 : i32
        %ki1_171 = arith.constant 1 : i32
        %ki2_171 = arith.constant 5 : i32
        %pad1_171 = arith.constant 1 : i32
        %outidx_171 = arith.constant 171 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_171, %smean_171, %sscale_171, %dilation_171, %padding_171, %ki0_171, %ki1_171, %ki2_171, %pad1_171, %out_v, %outidx_171) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 172: dilation=14 padding=56 kernel=(0,1,6) padding1=0
        %fidx_172 = arith.constant 172 : index
        %bidx_172 = arith.constant 423 : index
        %smidx_172 = arith.constant 843 : index
        %ssidx_172 = arith.constant 1263 : index
        %bias_172 = memref.load %in_v[%bidx_172] : memref<1511xi32>
        %smean_172 = memref.load %in_v[%smidx_172] : memref<1511xi32>
        %sscale_172 = memref.load %in_v[%ssidx_172] : memref<1511xi32>
        %dilation_172 = arith.constant 14 : i32
        %padding_172 = arith.constant 56 : i32
        %ki0_172 = arith.constant 0 : i32
        %ki1_172 = arith.constant 1 : i32
        %ki2_172 = arith.constant 6 : i32
        %pad1_172 = arith.constant 0 : i32
        %outidx_172 = arith.constant 172 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_172, %smean_172, %sscale_172, %dilation_172, %padding_172, %ki0_172, %ki1_172, %ki2_172, %pad1_172, %out_v, %outidx_172) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 173: dilation=14 padding=56 kernel=(0,1,7) padding1=1
        %fidx_173 = arith.constant 173 : index
        %bidx_173 = arith.constant 424 : index
        %smidx_173 = arith.constant 844 : index
        %ssidx_173 = arith.constant 1264 : index
        %bias_173 = memref.load %in_v[%bidx_173] : memref<1511xi32>
        %smean_173 = memref.load %in_v[%smidx_173] : memref<1511xi32>
        %sscale_173 = memref.load %in_v[%ssidx_173] : memref<1511xi32>
        %dilation_173 = arith.constant 14 : i32
        %padding_173 = arith.constant 56 : i32
        %ki0_173 = arith.constant 0 : i32
        %ki1_173 = arith.constant 1 : i32
        %ki2_173 = arith.constant 7 : i32
        %pad1_173 = arith.constant 1 : i32
        %outidx_173 = arith.constant 173 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_173, %smean_173, %sscale_173, %dilation_173, %padding_173, %ki0_173, %ki1_173, %ki2_173, %pad1_173, %out_v, %outidx_173) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 174: dilation=14 padding=56 kernel=(0,1,8) padding1=0
        %fidx_174 = arith.constant 174 : index
        %bidx_174 = arith.constant 425 : index
        %smidx_174 = arith.constant 845 : index
        %ssidx_174 = arith.constant 1265 : index
        %bias_174 = memref.load %in_v[%bidx_174] : memref<1511xi32>
        %smean_174 = memref.load %in_v[%smidx_174] : memref<1511xi32>
        %sscale_174 = memref.load %in_v[%ssidx_174] : memref<1511xi32>
        %dilation_174 = arith.constant 14 : i32
        %padding_174 = arith.constant 56 : i32
        %ki0_174 = arith.constant 0 : i32
        %ki1_174 = arith.constant 1 : i32
        %ki2_174 = arith.constant 8 : i32
        %pad1_174 = arith.constant 0 : i32
        %outidx_174 = arith.constant 174 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_174, %smean_174, %sscale_174, %dilation_174, %padding_174, %ki0_174, %ki1_174, %ki2_174, %pad1_174, %out_v, %outidx_174) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 175: dilation=14 padding=56 kernel=(0,2,3) padding1=1
        %fidx_175 = arith.constant 175 : index
        %bidx_175 = arith.constant 426 : index
        %smidx_175 = arith.constant 846 : index
        %ssidx_175 = arith.constant 1266 : index
        %bias_175 = memref.load %in_v[%bidx_175] : memref<1511xi32>
        %smean_175 = memref.load %in_v[%smidx_175] : memref<1511xi32>
        %sscale_175 = memref.load %in_v[%ssidx_175] : memref<1511xi32>
        %dilation_175 = arith.constant 14 : i32
        %padding_175 = arith.constant 56 : i32
        %ki0_175 = arith.constant 0 : i32
        %ki1_175 = arith.constant 2 : i32
        %ki2_175 = arith.constant 3 : i32
        %pad1_175 = arith.constant 1 : i32
        %outidx_175 = arith.constant 175 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_175, %smean_175, %sscale_175, %dilation_175, %padding_175, %ki0_175, %ki1_175, %ki2_175, %pad1_175, %out_v, %outidx_175) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 176: dilation=14 padding=56 kernel=(0,2,4) padding1=0
        %fidx_176 = arith.constant 176 : index
        %bidx_176 = arith.constant 427 : index
        %smidx_176 = arith.constant 847 : index
        %ssidx_176 = arith.constant 1267 : index
        %bias_176 = memref.load %in_v[%bidx_176] : memref<1511xi32>
        %smean_176 = memref.load %in_v[%smidx_176] : memref<1511xi32>
        %sscale_176 = memref.load %in_v[%ssidx_176] : memref<1511xi32>
        %dilation_176 = arith.constant 14 : i32
        %padding_176 = arith.constant 56 : i32
        %ki0_176 = arith.constant 0 : i32
        %ki1_176 = arith.constant 2 : i32
        %ki2_176 = arith.constant 4 : i32
        %pad1_176 = arith.constant 0 : i32
        %outidx_176 = arith.constant 176 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_176, %smean_176, %sscale_176, %dilation_176, %padding_176, %ki0_176, %ki1_176, %ki2_176, %pad1_176, %out_v, %outidx_176) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 177: dilation=14 padding=56 kernel=(0,2,5) padding1=1
        %fidx_177 = arith.constant 177 : index
        %bidx_177 = arith.constant 428 : index
        %smidx_177 = arith.constant 848 : index
        %ssidx_177 = arith.constant 1268 : index
        %bias_177 = memref.load %in_v[%bidx_177] : memref<1511xi32>
        %smean_177 = memref.load %in_v[%smidx_177] : memref<1511xi32>
        %sscale_177 = memref.load %in_v[%ssidx_177] : memref<1511xi32>
        %dilation_177 = arith.constant 14 : i32
        %padding_177 = arith.constant 56 : i32
        %ki0_177 = arith.constant 0 : i32
        %ki1_177 = arith.constant 2 : i32
        %ki2_177 = arith.constant 5 : i32
        %pad1_177 = arith.constant 1 : i32
        %outidx_177 = arith.constant 177 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_177, %smean_177, %sscale_177, %dilation_177, %padding_177, %ki0_177, %ki1_177, %ki2_177, %pad1_177, %out_v, %outidx_177) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 178: dilation=14 padding=56 kernel=(0,2,6) padding1=0
        %fidx_178 = arith.constant 178 : index
        %bidx_178 = arith.constant 429 : index
        %smidx_178 = arith.constant 849 : index
        %ssidx_178 = arith.constant 1269 : index
        %bias_178 = memref.load %in_v[%bidx_178] : memref<1511xi32>
        %smean_178 = memref.load %in_v[%smidx_178] : memref<1511xi32>
        %sscale_178 = memref.load %in_v[%ssidx_178] : memref<1511xi32>
        %dilation_178 = arith.constant 14 : i32
        %padding_178 = arith.constant 56 : i32
        %ki0_178 = arith.constant 0 : i32
        %ki1_178 = arith.constant 2 : i32
        %ki2_178 = arith.constant 6 : i32
        %pad1_178 = arith.constant 0 : i32
        %outidx_178 = arith.constant 178 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_178, %smean_178, %sscale_178, %dilation_178, %padding_178, %ki0_178, %ki1_178, %ki2_178, %pad1_178, %out_v, %outidx_178) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 179: dilation=14 padding=56 kernel=(0,2,7) padding1=1
        %fidx_179 = arith.constant 179 : index
        %bidx_179 = arith.constant 430 : index
        %smidx_179 = arith.constant 850 : index
        %ssidx_179 = arith.constant 1270 : index
        %bias_179 = memref.load %in_v[%bidx_179] : memref<1511xi32>
        %smean_179 = memref.load %in_v[%smidx_179] : memref<1511xi32>
        %sscale_179 = memref.load %in_v[%ssidx_179] : memref<1511xi32>
        %dilation_179 = arith.constant 14 : i32
        %padding_179 = arith.constant 56 : i32
        %ki0_179 = arith.constant 0 : i32
        %ki1_179 = arith.constant 2 : i32
        %ki2_179 = arith.constant 7 : i32
        %pad1_179 = arith.constant 1 : i32
        %outidx_179 = arith.constant 179 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_179, %smean_179, %sscale_179, %dilation_179, %padding_179, %ki0_179, %ki1_179, %ki2_179, %pad1_179, %out_v, %outidx_179) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 180: dilation=14 padding=56 kernel=(0,2,8) padding1=0
        %fidx_180 = arith.constant 180 : index
        %bidx_180 = arith.constant 431 : index
        %smidx_180 = arith.constant 851 : index
        %ssidx_180 = arith.constant 1271 : index
        %bias_180 = memref.load %in_v[%bidx_180] : memref<1511xi32>
        %smean_180 = memref.load %in_v[%smidx_180] : memref<1511xi32>
        %sscale_180 = memref.load %in_v[%ssidx_180] : memref<1511xi32>
        %dilation_180 = arith.constant 14 : i32
        %padding_180 = arith.constant 56 : i32
        %ki0_180 = arith.constant 0 : i32
        %ki1_180 = arith.constant 2 : i32
        %ki2_180 = arith.constant 8 : i32
        %pad1_180 = arith.constant 0 : i32
        %outidx_180 = arith.constant 180 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_180, %smean_180, %sscale_180, %dilation_180, %padding_180, %ki0_180, %ki1_180, %ki2_180, %pad1_180, %out_v, %outidx_180) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 181: dilation=14 padding=56 kernel=(0,3,4) padding1=1
        %fidx_181 = arith.constant 181 : index
        %bidx_181 = arith.constant 432 : index
        %smidx_181 = arith.constant 852 : index
        %ssidx_181 = arith.constant 1272 : index
        %bias_181 = memref.load %in_v[%bidx_181] : memref<1511xi32>
        %smean_181 = memref.load %in_v[%smidx_181] : memref<1511xi32>
        %sscale_181 = memref.load %in_v[%ssidx_181] : memref<1511xi32>
        %dilation_181 = arith.constant 14 : i32
        %padding_181 = arith.constant 56 : i32
        %ki0_181 = arith.constant 0 : i32
        %ki1_181 = arith.constant 3 : i32
        %ki2_181 = arith.constant 4 : i32
        %pad1_181 = arith.constant 1 : i32
        %outidx_181 = arith.constant 181 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_181, %smean_181, %sscale_181, %dilation_181, %padding_181, %ki0_181, %ki1_181, %ki2_181, %pad1_181, %out_v, %outidx_181) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 182: dilation=14 padding=56 kernel=(0,3,5) padding1=0
        %fidx_182 = arith.constant 182 : index
        %bidx_182 = arith.constant 433 : index
        %smidx_182 = arith.constant 853 : index
        %ssidx_182 = arith.constant 1273 : index
        %bias_182 = memref.load %in_v[%bidx_182] : memref<1511xi32>
        %smean_182 = memref.load %in_v[%smidx_182] : memref<1511xi32>
        %sscale_182 = memref.load %in_v[%ssidx_182] : memref<1511xi32>
        %dilation_182 = arith.constant 14 : i32
        %padding_182 = arith.constant 56 : i32
        %ki0_182 = arith.constant 0 : i32
        %ki1_182 = arith.constant 3 : i32
        %ki2_182 = arith.constant 5 : i32
        %pad1_182 = arith.constant 0 : i32
        %outidx_182 = arith.constant 182 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_182, %smean_182, %sscale_182, %dilation_182, %padding_182, %ki0_182, %ki1_182, %ki2_182, %pad1_182, %out_v, %outidx_182) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 183: dilation=14 padding=56 kernel=(0,3,6) padding1=1
        %fidx_183 = arith.constant 183 : index
        %bidx_183 = arith.constant 434 : index
        %smidx_183 = arith.constant 854 : index
        %ssidx_183 = arith.constant 1274 : index
        %bias_183 = memref.load %in_v[%bidx_183] : memref<1511xi32>
        %smean_183 = memref.load %in_v[%smidx_183] : memref<1511xi32>
        %sscale_183 = memref.load %in_v[%ssidx_183] : memref<1511xi32>
        %dilation_183 = arith.constant 14 : i32
        %padding_183 = arith.constant 56 : i32
        %ki0_183 = arith.constant 0 : i32
        %ki1_183 = arith.constant 3 : i32
        %ki2_183 = arith.constant 6 : i32
        %pad1_183 = arith.constant 1 : i32
        %outidx_183 = arith.constant 183 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_183, %smean_183, %sscale_183, %dilation_183, %padding_183, %ki0_183, %ki1_183, %ki2_183, %pad1_183, %out_v, %outidx_183) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 184: dilation=14 padding=56 kernel=(0,3,7) padding1=0
        %fidx_184 = arith.constant 184 : index
        %bidx_184 = arith.constant 435 : index
        %smidx_184 = arith.constant 855 : index
        %ssidx_184 = arith.constant 1275 : index
        %bias_184 = memref.load %in_v[%bidx_184] : memref<1511xi32>
        %smean_184 = memref.load %in_v[%smidx_184] : memref<1511xi32>
        %sscale_184 = memref.load %in_v[%ssidx_184] : memref<1511xi32>
        %dilation_184 = arith.constant 14 : i32
        %padding_184 = arith.constant 56 : i32
        %ki0_184 = arith.constant 0 : i32
        %ki1_184 = arith.constant 3 : i32
        %ki2_184 = arith.constant 7 : i32
        %pad1_184 = arith.constant 0 : i32
        %outidx_184 = arith.constant 184 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_184, %smean_184, %sscale_184, %dilation_184, %padding_184, %ki0_184, %ki1_184, %ki2_184, %pad1_184, %out_v, %outidx_184) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 185: dilation=14 padding=56 kernel=(0,3,8) padding1=1
        %fidx_185 = arith.constant 185 : index
        %bidx_185 = arith.constant 436 : index
        %smidx_185 = arith.constant 856 : index
        %ssidx_185 = arith.constant 1276 : index
        %bias_185 = memref.load %in_v[%bidx_185] : memref<1511xi32>
        %smean_185 = memref.load %in_v[%smidx_185] : memref<1511xi32>
        %sscale_185 = memref.load %in_v[%ssidx_185] : memref<1511xi32>
        %dilation_185 = arith.constant 14 : i32
        %padding_185 = arith.constant 56 : i32
        %ki0_185 = arith.constant 0 : i32
        %ki1_185 = arith.constant 3 : i32
        %ki2_185 = arith.constant 8 : i32
        %pad1_185 = arith.constant 1 : i32
        %outidx_185 = arith.constant 185 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_185, %smean_185, %sscale_185, %dilation_185, %padding_185, %ki0_185, %ki1_185, %ki2_185, %pad1_185, %out_v, %outidx_185) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 186: dilation=14 padding=56 kernel=(0,4,5) padding1=0
        %fidx_186 = arith.constant 186 : index
        %bidx_186 = arith.constant 437 : index
        %smidx_186 = arith.constant 857 : index
        %ssidx_186 = arith.constant 1277 : index
        %bias_186 = memref.load %in_v[%bidx_186] : memref<1511xi32>
        %smean_186 = memref.load %in_v[%smidx_186] : memref<1511xi32>
        %sscale_186 = memref.load %in_v[%ssidx_186] : memref<1511xi32>
        %dilation_186 = arith.constant 14 : i32
        %padding_186 = arith.constant 56 : i32
        %ki0_186 = arith.constant 0 : i32
        %ki1_186 = arith.constant 4 : i32
        %ki2_186 = arith.constant 5 : i32
        %pad1_186 = arith.constant 0 : i32
        %outidx_186 = arith.constant 186 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_186, %smean_186, %sscale_186, %dilation_186, %padding_186, %ki0_186, %ki1_186, %ki2_186, %pad1_186, %out_v, %outidx_186) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 187: dilation=14 padding=56 kernel=(0,4,6) padding1=1
        %fidx_187 = arith.constant 187 : index
        %bidx_187 = arith.constant 438 : index
        %smidx_187 = arith.constant 858 : index
        %ssidx_187 = arith.constant 1278 : index
        %bias_187 = memref.load %in_v[%bidx_187] : memref<1511xi32>
        %smean_187 = memref.load %in_v[%smidx_187] : memref<1511xi32>
        %sscale_187 = memref.load %in_v[%ssidx_187] : memref<1511xi32>
        %dilation_187 = arith.constant 14 : i32
        %padding_187 = arith.constant 56 : i32
        %ki0_187 = arith.constant 0 : i32
        %ki1_187 = arith.constant 4 : i32
        %ki2_187 = arith.constant 6 : i32
        %pad1_187 = arith.constant 1 : i32
        %outidx_187 = arith.constant 187 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_187, %smean_187, %sscale_187, %dilation_187, %padding_187, %ki0_187, %ki1_187, %ki2_187, %pad1_187, %out_v, %outidx_187) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 188: dilation=14 padding=56 kernel=(0,4,7) padding1=0
        %fidx_188 = arith.constant 188 : index
        %bidx_188 = arith.constant 439 : index
        %smidx_188 = arith.constant 859 : index
        %ssidx_188 = arith.constant 1279 : index
        %bias_188 = memref.load %in_v[%bidx_188] : memref<1511xi32>
        %smean_188 = memref.load %in_v[%smidx_188] : memref<1511xi32>
        %sscale_188 = memref.load %in_v[%ssidx_188] : memref<1511xi32>
        %dilation_188 = arith.constant 14 : i32
        %padding_188 = arith.constant 56 : i32
        %ki0_188 = arith.constant 0 : i32
        %ki1_188 = arith.constant 4 : i32
        %ki2_188 = arith.constant 7 : i32
        %pad1_188 = arith.constant 0 : i32
        %outidx_188 = arith.constant 188 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_188, %smean_188, %sscale_188, %dilation_188, %padding_188, %ki0_188, %ki1_188, %ki2_188, %pad1_188, %out_v, %outidx_188) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 189: dilation=14 padding=56 kernel=(0,4,8) padding1=1
        %fidx_189 = arith.constant 189 : index
        %bidx_189 = arith.constant 440 : index
        %smidx_189 = arith.constant 860 : index
        %ssidx_189 = arith.constant 1280 : index
        %bias_189 = memref.load %in_v[%bidx_189] : memref<1511xi32>
        %smean_189 = memref.load %in_v[%smidx_189] : memref<1511xi32>
        %sscale_189 = memref.load %in_v[%ssidx_189] : memref<1511xi32>
        %dilation_189 = arith.constant 14 : i32
        %padding_189 = arith.constant 56 : i32
        %ki0_189 = arith.constant 0 : i32
        %ki1_189 = arith.constant 4 : i32
        %ki2_189 = arith.constant 8 : i32
        %pad1_189 = arith.constant 1 : i32
        %outidx_189 = arith.constant 189 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_189, %smean_189, %sscale_189, %dilation_189, %padding_189, %ki0_189, %ki1_189, %ki2_189, %pad1_189, %out_v, %outidx_189) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 190: dilation=14 padding=56 kernel=(0,5,6) padding1=0
        %fidx_190 = arith.constant 190 : index
        %bidx_190 = arith.constant 441 : index
        %smidx_190 = arith.constant 861 : index
        %ssidx_190 = arith.constant 1281 : index
        %bias_190 = memref.load %in_v[%bidx_190] : memref<1511xi32>
        %smean_190 = memref.load %in_v[%smidx_190] : memref<1511xi32>
        %sscale_190 = memref.load %in_v[%ssidx_190] : memref<1511xi32>
        %dilation_190 = arith.constant 14 : i32
        %padding_190 = arith.constant 56 : i32
        %ki0_190 = arith.constant 0 : i32
        %ki1_190 = arith.constant 5 : i32
        %ki2_190 = arith.constant 6 : i32
        %pad1_190 = arith.constant 0 : i32
        %outidx_190 = arith.constant 190 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_190, %smean_190, %sscale_190, %dilation_190, %padding_190, %ki0_190, %ki1_190, %ki2_190, %pad1_190, %out_v, %outidx_190) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 191: dilation=14 padding=56 kernel=(0,5,7) padding1=1
        %fidx_191 = arith.constant 191 : index
        %bidx_191 = arith.constant 442 : index
        %smidx_191 = arith.constant 862 : index
        %ssidx_191 = arith.constant 1282 : index
        %bias_191 = memref.load %in_v[%bidx_191] : memref<1511xi32>
        %smean_191 = memref.load %in_v[%smidx_191] : memref<1511xi32>
        %sscale_191 = memref.load %in_v[%ssidx_191] : memref<1511xi32>
        %dilation_191 = arith.constant 14 : i32
        %padding_191 = arith.constant 56 : i32
        %ki0_191 = arith.constant 0 : i32
        %ki1_191 = arith.constant 5 : i32
        %ki2_191 = arith.constant 7 : i32
        %pad1_191 = arith.constant 1 : i32
        %outidx_191 = arith.constant 191 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_191, %smean_191, %sscale_191, %dilation_191, %padding_191, %ki0_191, %ki1_191, %ki2_191, %pad1_191, %out_v, %outidx_191) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 192: dilation=14 padding=56 kernel=(0,5,8) padding1=0
        %fidx_192 = arith.constant 192 : index
        %bidx_192 = arith.constant 443 : index
        %smidx_192 = arith.constant 863 : index
        %ssidx_192 = arith.constant 1283 : index
        %bias_192 = memref.load %in_v[%bidx_192] : memref<1511xi32>
        %smean_192 = memref.load %in_v[%smidx_192] : memref<1511xi32>
        %sscale_192 = memref.load %in_v[%ssidx_192] : memref<1511xi32>
        %dilation_192 = arith.constant 14 : i32
        %padding_192 = arith.constant 56 : i32
        %ki0_192 = arith.constant 0 : i32
        %ki1_192 = arith.constant 5 : i32
        %ki2_192 = arith.constant 8 : i32
        %pad1_192 = arith.constant 0 : i32
        %outidx_192 = arith.constant 192 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_192, %smean_192, %sscale_192, %dilation_192, %padding_192, %ki0_192, %ki1_192, %ki2_192, %pad1_192, %out_v, %outidx_192) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 193: dilation=14 padding=56 kernel=(0,6,7) padding1=1
        %fidx_193 = arith.constant 193 : index
        %bidx_193 = arith.constant 444 : index
        %smidx_193 = arith.constant 864 : index
        %ssidx_193 = arith.constant 1284 : index
        %bias_193 = memref.load %in_v[%bidx_193] : memref<1511xi32>
        %smean_193 = memref.load %in_v[%smidx_193] : memref<1511xi32>
        %sscale_193 = memref.load %in_v[%ssidx_193] : memref<1511xi32>
        %dilation_193 = arith.constant 14 : i32
        %padding_193 = arith.constant 56 : i32
        %ki0_193 = arith.constant 0 : i32
        %ki1_193 = arith.constant 6 : i32
        %ki2_193 = arith.constant 7 : i32
        %pad1_193 = arith.constant 1 : i32
        %outidx_193 = arith.constant 193 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_193, %smean_193, %sscale_193, %dilation_193, %padding_193, %ki0_193, %ki1_193, %ki2_193, %pad1_193, %out_v, %outidx_193) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 194: dilation=14 padding=56 kernel=(0,6,8) padding1=0
        %fidx_194 = arith.constant 194 : index
        %bidx_194 = arith.constant 445 : index
        %smidx_194 = arith.constant 865 : index
        %ssidx_194 = arith.constant 1285 : index
        %bias_194 = memref.load %in_v[%bidx_194] : memref<1511xi32>
        %smean_194 = memref.load %in_v[%smidx_194] : memref<1511xi32>
        %sscale_194 = memref.load %in_v[%ssidx_194] : memref<1511xi32>
        %dilation_194 = arith.constant 14 : i32
        %padding_194 = arith.constant 56 : i32
        %ki0_194 = arith.constant 0 : i32
        %ki1_194 = arith.constant 6 : i32
        %ki2_194 = arith.constant 8 : i32
        %pad1_194 = arith.constant 0 : i32
        %outidx_194 = arith.constant 194 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_194, %smean_194, %sscale_194, %dilation_194, %padding_194, %ki0_194, %ki1_194, %ki2_194, %pad1_194, %out_v, %outidx_194) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 195: dilation=14 padding=56 kernel=(0,7,8) padding1=1
        %fidx_195 = arith.constant 195 : index
        %bidx_195 = arith.constant 446 : index
        %smidx_195 = arith.constant 866 : index
        %ssidx_195 = arith.constant 1286 : index
        %bias_195 = memref.load %in_v[%bidx_195] : memref<1511xi32>
        %smean_195 = memref.load %in_v[%smidx_195] : memref<1511xi32>
        %sscale_195 = memref.load %in_v[%ssidx_195] : memref<1511xi32>
        %dilation_195 = arith.constant 14 : i32
        %padding_195 = arith.constant 56 : i32
        %ki0_195 = arith.constant 0 : i32
        %ki1_195 = arith.constant 7 : i32
        %ki2_195 = arith.constant 8 : i32
        %pad1_195 = arith.constant 1 : i32
        %outidx_195 = arith.constant 195 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_195, %smean_195, %sscale_195, %dilation_195, %padding_195, %ki0_195, %ki1_195, %ki2_195, %pad1_195, %out_v, %outidx_195) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 196: dilation=14 padding=56 kernel=(1,2,3) padding1=0
        %fidx_196 = arith.constant 196 : index
        %bidx_196 = arith.constant 447 : index
        %smidx_196 = arith.constant 867 : index
        %ssidx_196 = arith.constant 1287 : index
        %bias_196 = memref.load %in_v[%bidx_196] : memref<1511xi32>
        %smean_196 = memref.load %in_v[%smidx_196] : memref<1511xi32>
        %sscale_196 = memref.load %in_v[%ssidx_196] : memref<1511xi32>
        %dilation_196 = arith.constant 14 : i32
        %padding_196 = arith.constant 56 : i32
        %ki0_196 = arith.constant 1 : i32
        %ki1_196 = arith.constant 2 : i32
        %ki2_196 = arith.constant 3 : i32
        %pad1_196 = arith.constant 0 : i32
        %outidx_196 = arith.constant 196 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_196, %smean_196, %sscale_196, %dilation_196, %padding_196, %ki0_196, %ki1_196, %ki2_196, %pad1_196, %out_v, %outidx_196) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 197: dilation=14 padding=56 kernel=(1,2,4) padding1=1
        %fidx_197 = arith.constant 197 : index
        %bidx_197 = arith.constant 448 : index
        %smidx_197 = arith.constant 868 : index
        %ssidx_197 = arith.constant 1288 : index
        %bias_197 = memref.load %in_v[%bidx_197] : memref<1511xi32>
        %smean_197 = memref.load %in_v[%smidx_197] : memref<1511xi32>
        %sscale_197 = memref.load %in_v[%ssidx_197] : memref<1511xi32>
        %dilation_197 = arith.constant 14 : i32
        %padding_197 = arith.constant 56 : i32
        %ki0_197 = arith.constant 1 : i32
        %ki1_197 = arith.constant 2 : i32
        %ki2_197 = arith.constant 4 : i32
        %pad1_197 = arith.constant 1 : i32
        %outidx_197 = arith.constant 197 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_197, %smean_197, %sscale_197, %dilation_197, %padding_197, %ki0_197, %ki1_197, %ki2_197, %pad1_197, %out_v, %outidx_197) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 198: dilation=14 padding=56 kernel=(1,2,5) padding1=0
        %fidx_198 = arith.constant 198 : index
        %bidx_198 = arith.constant 449 : index
        %smidx_198 = arith.constant 869 : index
        %ssidx_198 = arith.constant 1289 : index
        %bias_198 = memref.load %in_v[%bidx_198] : memref<1511xi32>
        %smean_198 = memref.load %in_v[%smidx_198] : memref<1511xi32>
        %sscale_198 = memref.load %in_v[%ssidx_198] : memref<1511xi32>
        %dilation_198 = arith.constant 14 : i32
        %padding_198 = arith.constant 56 : i32
        %ki0_198 = arith.constant 1 : i32
        %ki1_198 = arith.constant 2 : i32
        %ki2_198 = arith.constant 5 : i32
        %pad1_198 = arith.constant 0 : i32
        %outidx_198 = arith.constant 198 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_198, %smean_198, %sscale_198, %dilation_198, %padding_198, %ki0_198, %ki1_198, %ki2_198, %pad1_198, %out_v, %outidx_198) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 199: dilation=14 padding=56 kernel=(1,2,6) padding1=1
        %fidx_199 = arith.constant 199 : index
        %bidx_199 = arith.constant 450 : index
        %smidx_199 = arith.constant 870 : index
        %ssidx_199 = arith.constant 1290 : index
        %bias_199 = memref.load %in_v[%bidx_199] : memref<1511xi32>
        %smean_199 = memref.load %in_v[%smidx_199] : memref<1511xi32>
        %sscale_199 = memref.load %in_v[%ssidx_199] : memref<1511xi32>
        %dilation_199 = arith.constant 14 : i32
        %padding_199 = arith.constant 56 : i32
        %ki0_199 = arith.constant 1 : i32
        %ki1_199 = arith.constant 2 : i32
        %ki2_199 = arith.constant 6 : i32
        %pad1_199 = arith.constant 1 : i32
        %outidx_199 = arith.constant 199 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_199, %smean_199, %sscale_199, %dilation_199, %padding_199, %ki0_199, %ki1_199, %ki2_199, %pad1_199, %out_v, %outidx_199) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 200: dilation=14 padding=56 kernel=(1,2,7) padding1=0
        %fidx_200 = arith.constant 200 : index
        %bidx_200 = arith.constant 451 : index
        %smidx_200 = arith.constant 871 : index
        %ssidx_200 = arith.constant 1291 : index
        %bias_200 = memref.load %in_v[%bidx_200] : memref<1511xi32>
        %smean_200 = memref.load %in_v[%smidx_200] : memref<1511xi32>
        %sscale_200 = memref.load %in_v[%ssidx_200] : memref<1511xi32>
        %dilation_200 = arith.constant 14 : i32
        %padding_200 = arith.constant 56 : i32
        %ki0_200 = arith.constant 1 : i32
        %ki1_200 = arith.constant 2 : i32
        %ki2_200 = arith.constant 7 : i32
        %pad1_200 = arith.constant 0 : i32
        %outidx_200 = arith.constant 200 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_200, %smean_200, %sscale_200, %dilation_200, %padding_200, %ki0_200, %ki1_200, %ki2_200, %pad1_200, %out_v, %outidx_200) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 201: dilation=14 padding=56 kernel=(1,2,8) padding1=1
        %fidx_201 = arith.constant 201 : index
        %bidx_201 = arith.constant 452 : index
        %smidx_201 = arith.constant 872 : index
        %ssidx_201 = arith.constant 1292 : index
        %bias_201 = memref.load %in_v[%bidx_201] : memref<1511xi32>
        %smean_201 = memref.load %in_v[%smidx_201] : memref<1511xi32>
        %sscale_201 = memref.load %in_v[%ssidx_201] : memref<1511xi32>
        %dilation_201 = arith.constant 14 : i32
        %padding_201 = arith.constant 56 : i32
        %ki0_201 = arith.constant 1 : i32
        %ki1_201 = arith.constant 2 : i32
        %ki2_201 = arith.constant 8 : i32
        %pad1_201 = arith.constant 1 : i32
        %outidx_201 = arith.constant 201 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_201, %smean_201, %sscale_201, %dilation_201, %padding_201, %ki0_201, %ki1_201, %ki2_201, %pad1_201, %out_v, %outidx_201) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 202: dilation=14 padding=56 kernel=(1,3,4) padding1=0
        %fidx_202 = arith.constant 202 : index
        %bidx_202 = arith.constant 453 : index
        %smidx_202 = arith.constant 873 : index
        %ssidx_202 = arith.constant 1293 : index
        %bias_202 = memref.load %in_v[%bidx_202] : memref<1511xi32>
        %smean_202 = memref.load %in_v[%smidx_202] : memref<1511xi32>
        %sscale_202 = memref.load %in_v[%ssidx_202] : memref<1511xi32>
        %dilation_202 = arith.constant 14 : i32
        %padding_202 = arith.constant 56 : i32
        %ki0_202 = arith.constant 1 : i32
        %ki1_202 = arith.constant 3 : i32
        %ki2_202 = arith.constant 4 : i32
        %pad1_202 = arith.constant 0 : i32
        %outidx_202 = arith.constant 202 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_202, %smean_202, %sscale_202, %dilation_202, %padding_202, %ki0_202, %ki1_202, %ki2_202, %pad1_202, %out_v, %outidx_202) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 203: dilation=14 padding=56 kernel=(1,3,5) padding1=1
        %fidx_203 = arith.constant 203 : index
        %bidx_203 = arith.constant 454 : index
        %smidx_203 = arith.constant 874 : index
        %ssidx_203 = arith.constant 1294 : index
        %bias_203 = memref.load %in_v[%bidx_203] : memref<1511xi32>
        %smean_203 = memref.load %in_v[%smidx_203] : memref<1511xi32>
        %sscale_203 = memref.load %in_v[%ssidx_203] : memref<1511xi32>
        %dilation_203 = arith.constant 14 : i32
        %padding_203 = arith.constant 56 : i32
        %ki0_203 = arith.constant 1 : i32
        %ki1_203 = arith.constant 3 : i32
        %ki2_203 = arith.constant 5 : i32
        %pad1_203 = arith.constant 1 : i32
        %outidx_203 = arith.constant 203 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_203, %smean_203, %sscale_203, %dilation_203, %padding_203, %ki0_203, %ki1_203, %ki2_203, %pad1_203, %out_v, %outidx_203) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 204: dilation=14 padding=56 kernel=(1,3,6) padding1=0
        %fidx_204 = arith.constant 204 : index
        %bidx_204 = arith.constant 455 : index
        %smidx_204 = arith.constant 875 : index
        %ssidx_204 = arith.constant 1295 : index
        %bias_204 = memref.load %in_v[%bidx_204] : memref<1511xi32>
        %smean_204 = memref.load %in_v[%smidx_204] : memref<1511xi32>
        %sscale_204 = memref.load %in_v[%ssidx_204] : memref<1511xi32>
        %dilation_204 = arith.constant 14 : i32
        %padding_204 = arith.constant 56 : i32
        %ki0_204 = arith.constant 1 : i32
        %ki1_204 = arith.constant 3 : i32
        %ki2_204 = arith.constant 6 : i32
        %pad1_204 = arith.constant 0 : i32
        %outidx_204 = arith.constant 204 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_204, %smean_204, %sscale_204, %dilation_204, %padding_204, %ki0_204, %ki1_204, %ki2_204, %pad1_204, %out_v, %outidx_204) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 205: dilation=14 padding=56 kernel=(1,3,7) padding1=1
        %fidx_205 = arith.constant 205 : index
        %bidx_205 = arith.constant 456 : index
        %smidx_205 = arith.constant 876 : index
        %ssidx_205 = arith.constant 1296 : index
        %bias_205 = memref.load %in_v[%bidx_205] : memref<1511xi32>
        %smean_205 = memref.load %in_v[%smidx_205] : memref<1511xi32>
        %sscale_205 = memref.load %in_v[%ssidx_205] : memref<1511xi32>
        %dilation_205 = arith.constant 14 : i32
        %padding_205 = arith.constant 56 : i32
        %ki0_205 = arith.constant 1 : i32
        %ki1_205 = arith.constant 3 : i32
        %ki2_205 = arith.constant 7 : i32
        %pad1_205 = arith.constant 1 : i32
        %outidx_205 = arith.constant 205 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_205, %smean_205, %sscale_205, %dilation_205, %padding_205, %ki0_205, %ki1_205, %ki2_205, %pad1_205, %out_v, %outidx_205) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 206: dilation=14 padding=56 kernel=(1,3,8) padding1=0
        %fidx_206 = arith.constant 206 : index
        %bidx_206 = arith.constant 457 : index
        %smidx_206 = arith.constant 877 : index
        %ssidx_206 = arith.constant 1297 : index
        %bias_206 = memref.load %in_v[%bidx_206] : memref<1511xi32>
        %smean_206 = memref.load %in_v[%smidx_206] : memref<1511xi32>
        %sscale_206 = memref.load %in_v[%ssidx_206] : memref<1511xi32>
        %dilation_206 = arith.constant 14 : i32
        %padding_206 = arith.constant 56 : i32
        %ki0_206 = arith.constant 1 : i32
        %ki1_206 = arith.constant 3 : i32
        %ki2_206 = arith.constant 8 : i32
        %pad1_206 = arith.constant 0 : i32
        %outidx_206 = arith.constant 206 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_206, %smean_206, %sscale_206, %dilation_206, %padding_206, %ki0_206, %ki1_206, %ki2_206, %pad1_206, %out_v, %outidx_206) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 207: dilation=14 padding=56 kernel=(1,4,5) padding1=1
        %fidx_207 = arith.constant 207 : index
        %bidx_207 = arith.constant 458 : index
        %smidx_207 = arith.constant 878 : index
        %ssidx_207 = arith.constant 1298 : index
        %bias_207 = memref.load %in_v[%bidx_207] : memref<1511xi32>
        %smean_207 = memref.load %in_v[%smidx_207] : memref<1511xi32>
        %sscale_207 = memref.load %in_v[%ssidx_207] : memref<1511xi32>
        %dilation_207 = arith.constant 14 : i32
        %padding_207 = arith.constant 56 : i32
        %ki0_207 = arith.constant 1 : i32
        %ki1_207 = arith.constant 4 : i32
        %ki2_207 = arith.constant 5 : i32
        %pad1_207 = arith.constant 1 : i32
        %outidx_207 = arith.constant 207 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_207, %smean_207, %sscale_207, %dilation_207, %padding_207, %ki0_207, %ki1_207, %ki2_207, %pad1_207, %out_v, %outidx_207) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 208: dilation=14 padding=56 kernel=(1,4,6) padding1=0
        %fidx_208 = arith.constant 208 : index
        %bidx_208 = arith.constant 459 : index
        %smidx_208 = arith.constant 879 : index
        %ssidx_208 = arith.constant 1299 : index
        %bias_208 = memref.load %in_v[%bidx_208] : memref<1511xi32>
        %smean_208 = memref.load %in_v[%smidx_208] : memref<1511xi32>
        %sscale_208 = memref.load %in_v[%ssidx_208] : memref<1511xi32>
        %dilation_208 = arith.constant 14 : i32
        %padding_208 = arith.constant 56 : i32
        %ki0_208 = arith.constant 1 : i32
        %ki1_208 = arith.constant 4 : i32
        %ki2_208 = arith.constant 6 : i32
        %pad1_208 = arith.constant 0 : i32
        %outidx_208 = arith.constant 208 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_208, %smean_208, %sscale_208, %dilation_208, %padding_208, %ki0_208, %ki1_208, %ki2_208, %pad1_208, %out_v, %outidx_208) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 209: dilation=14 padding=56 kernel=(1,4,7) padding1=1
        %fidx_209 = arith.constant 209 : index
        %bidx_209 = arith.constant 460 : index
        %smidx_209 = arith.constant 880 : index
        %ssidx_209 = arith.constant 1300 : index
        %bias_209 = memref.load %in_v[%bidx_209] : memref<1511xi32>
        %smean_209 = memref.load %in_v[%smidx_209] : memref<1511xi32>
        %sscale_209 = memref.load %in_v[%ssidx_209] : memref<1511xi32>
        %dilation_209 = arith.constant 14 : i32
        %padding_209 = arith.constant 56 : i32
        %ki0_209 = arith.constant 1 : i32
        %ki1_209 = arith.constant 4 : i32
        %ki2_209 = arith.constant 7 : i32
        %pad1_209 = arith.constant 1 : i32
        %outidx_209 = arith.constant 209 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_209, %smean_209, %sscale_209, %dilation_209, %padding_209, %ki0_209, %ki1_209, %ki2_209, %pad1_209, %out_v, %outidx_209) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 210: dilation=14 padding=56 kernel=(1,4,8) padding1=0
        %fidx_210 = arith.constant 210 : index
        %bidx_210 = arith.constant 461 : index
        %smidx_210 = arith.constant 881 : index
        %ssidx_210 = arith.constant 1301 : index
        %bias_210 = memref.load %in_v[%bidx_210] : memref<1511xi32>
        %smean_210 = memref.load %in_v[%smidx_210] : memref<1511xi32>
        %sscale_210 = memref.load %in_v[%ssidx_210] : memref<1511xi32>
        %dilation_210 = arith.constant 14 : i32
        %padding_210 = arith.constant 56 : i32
        %ki0_210 = arith.constant 1 : i32
        %ki1_210 = arith.constant 4 : i32
        %ki2_210 = arith.constant 8 : i32
        %pad1_210 = arith.constant 0 : i32
        %outidx_210 = arith.constant 210 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_210, %smean_210, %sscale_210, %dilation_210, %padding_210, %ki0_210, %ki1_210, %ki2_210, %pad1_210, %out_v, %outidx_210) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 211: dilation=14 padding=56 kernel=(1,5,6) padding1=1
        %fidx_211 = arith.constant 211 : index
        %bidx_211 = arith.constant 462 : index
        %smidx_211 = arith.constant 882 : index
        %ssidx_211 = arith.constant 1302 : index
        %bias_211 = memref.load %in_v[%bidx_211] : memref<1511xi32>
        %smean_211 = memref.load %in_v[%smidx_211] : memref<1511xi32>
        %sscale_211 = memref.load %in_v[%ssidx_211] : memref<1511xi32>
        %dilation_211 = arith.constant 14 : i32
        %padding_211 = arith.constant 56 : i32
        %ki0_211 = arith.constant 1 : i32
        %ki1_211 = arith.constant 5 : i32
        %ki2_211 = arith.constant 6 : i32
        %pad1_211 = arith.constant 1 : i32
        %outidx_211 = arith.constant 211 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_211, %smean_211, %sscale_211, %dilation_211, %padding_211, %ki0_211, %ki1_211, %ki2_211, %pad1_211, %out_v, %outidx_211) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 212: dilation=14 padding=56 kernel=(1,5,7) padding1=0
        %fidx_212 = arith.constant 212 : index
        %bidx_212 = arith.constant 463 : index
        %smidx_212 = arith.constant 883 : index
        %ssidx_212 = arith.constant 1303 : index
        %bias_212 = memref.load %in_v[%bidx_212] : memref<1511xi32>
        %smean_212 = memref.load %in_v[%smidx_212] : memref<1511xi32>
        %sscale_212 = memref.load %in_v[%ssidx_212] : memref<1511xi32>
        %dilation_212 = arith.constant 14 : i32
        %padding_212 = arith.constant 56 : i32
        %ki0_212 = arith.constant 1 : i32
        %ki1_212 = arith.constant 5 : i32
        %ki2_212 = arith.constant 7 : i32
        %pad1_212 = arith.constant 0 : i32
        %outidx_212 = arith.constant 212 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_212, %smean_212, %sscale_212, %dilation_212, %padding_212, %ki0_212, %ki1_212, %ki2_212, %pad1_212, %out_v, %outidx_212) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 213: dilation=14 padding=56 kernel=(1,5,8) padding1=1
        %fidx_213 = arith.constant 213 : index
        %bidx_213 = arith.constant 464 : index
        %smidx_213 = arith.constant 884 : index
        %ssidx_213 = arith.constant 1304 : index
        %bias_213 = memref.load %in_v[%bidx_213] : memref<1511xi32>
        %smean_213 = memref.load %in_v[%smidx_213] : memref<1511xi32>
        %sscale_213 = memref.load %in_v[%ssidx_213] : memref<1511xi32>
        %dilation_213 = arith.constant 14 : i32
        %padding_213 = arith.constant 56 : i32
        %ki0_213 = arith.constant 1 : i32
        %ki1_213 = arith.constant 5 : i32
        %ki2_213 = arith.constant 8 : i32
        %pad1_213 = arith.constant 1 : i32
        %outidx_213 = arith.constant 213 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_213, %smean_213, %sscale_213, %dilation_213, %padding_213, %ki0_213, %ki1_213, %ki2_213, %pad1_213, %out_v, %outidx_213) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 214: dilation=14 padding=56 kernel=(1,6,7) padding1=0
        %fidx_214 = arith.constant 214 : index
        %bidx_214 = arith.constant 465 : index
        %smidx_214 = arith.constant 885 : index
        %ssidx_214 = arith.constant 1305 : index
        %bias_214 = memref.load %in_v[%bidx_214] : memref<1511xi32>
        %smean_214 = memref.load %in_v[%smidx_214] : memref<1511xi32>
        %sscale_214 = memref.load %in_v[%ssidx_214] : memref<1511xi32>
        %dilation_214 = arith.constant 14 : i32
        %padding_214 = arith.constant 56 : i32
        %ki0_214 = arith.constant 1 : i32
        %ki1_214 = arith.constant 6 : i32
        %ki2_214 = arith.constant 7 : i32
        %pad1_214 = arith.constant 0 : i32
        %outidx_214 = arith.constant 214 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_214, %smean_214, %sscale_214, %dilation_214, %padding_214, %ki0_214, %ki1_214, %ki2_214, %pad1_214, %out_v, %outidx_214) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 215: dilation=14 padding=56 kernel=(1,6,8) padding1=1
        %fidx_215 = arith.constant 215 : index
        %bidx_215 = arith.constant 466 : index
        %smidx_215 = arith.constant 886 : index
        %ssidx_215 = arith.constant 1306 : index
        %bias_215 = memref.load %in_v[%bidx_215] : memref<1511xi32>
        %smean_215 = memref.load %in_v[%smidx_215] : memref<1511xi32>
        %sscale_215 = memref.load %in_v[%ssidx_215] : memref<1511xi32>
        %dilation_215 = arith.constant 14 : i32
        %padding_215 = arith.constant 56 : i32
        %ki0_215 = arith.constant 1 : i32
        %ki1_215 = arith.constant 6 : i32
        %ki2_215 = arith.constant 8 : i32
        %pad1_215 = arith.constant 1 : i32
        %outidx_215 = arith.constant 215 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_215, %smean_215, %sscale_215, %dilation_215, %padding_215, %ki0_215, %ki1_215, %ki2_215, %pad1_215, %out_v, %outidx_215) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 216: dilation=14 padding=56 kernel=(1,7,8) padding1=0
        %fidx_216 = arith.constant 216 : index
        %bidx_216 = arith.constant 467 : index
        %smidx_216 = arith.constant 887 : index
        %ssidx_216 = arith.constant 1307 : index
        %bias_216 = memref.load %in_v[%bidx_216] : memref<1511xi32>
        %smean_216 = memref.load %in_v[%smidx_216] : memref<1511xi32>
        %sscale_216 = memref.load %in_v[%ssidx_216] : memref<1511xi32>
        %dilation_216 = arith.constant 14 : i32
        %padding_216 = arith.constant 56 : i32
        %ki0_216 = arith.constant 1 : i32
        %ki1_216 = arith.constant 7 : i32
        %ki2_216 = arith.constant 8 : i32
        %pad1_216 = arith.constant 0 : i32
        %outidx_216 = arith.constant 216 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_216, %smean_216, %sscale_216, %dilation_216, %padding_216, %ki0_216, %ki1_216, %ki2_216, %pad1_216, %out_v, %outidx_216) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 217: dilation=14 padding=56 kernel=(2,3,4) padding1=1
        %fidx_217 = arith.constant 217 : index
        %bidx_217 = arith.constant 468 : index
        %smidx_217 = arith.constant 888 : index
        %ssidx_217 = arith.constant 1308 : index
        %bias_217 = memref.load %in_v[%bidx_217] : memref<1511xi32>
        %smean_217 = memref.load %in_v[%smidx_217] : memref<1511xi32>
        %sscale_217 = memref.load %in_v[%ssidx_217] : memref<1511xi32>
        %dilation_217 = arith.constant 14 : i32
        %padding_217 = arith.constant 56 : i32
        %ki0_217 = arith.constant 2 : i32
        %ki1_217 = arith.constant 3 : i32
        %ki2_217 = arith.constant 4 : i32
        %pad1_217 = arith.constant 1 : i32
        %outidx_217 = arith.constant 217 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_217, %smean_217, %sscale_217, %dilation_217, %padding_217, %ki0_217, %ki1_217, %ki2_217, %pad1_217, %out_v, %outidx_217) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 218: dilation=14 padding=56 kernel=(2,3,5) padding1=0
        %fidx_218 = arith.constant 218 : index
        %bidx_218 = arith.constant 469 : index
        %smidx_218 = arith.constant 889 : index
        %ssidx_218 = arith.constant 1309 : index
        %bias_218 = memref.load %in_v[%bidx_218] : memref<1511xi32>
        %smean_218 = memref.load %in_v[%smidx_218] : memref<1511xi32>
        %sscale_218 = memref.load %in_v[%ssidx_218] : memref<1511xi32>
        %dilation_218 = arith.constant 14 : i32
        %padding_218 = arith.constant 56 : i32
        %ki0_218 = arith.constant 2 : i32
        %ki1_218 = arith.constant 3 : i32
        %ki2_218 = arith.constant 5 : i32
        %pad1_218 = arith.constant 0 : i32
        %outidx_218 = arith.constant 218 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_218, %smean_218, %sscale_218, %dilation_218, %padding_218, %ki0_218, %ki1_218, %ki2_218, %pad1_218, %out_v, %outidx_218) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 219: dilation=14 padding=56 kernel=(2,3,6) padding1=1
        %fidx_219 = arith.constant 219 : index
        %bidx_219 = arith.constant 470 : index
        %smidx_219 = arith.constant 890 : index
        %ssidx_219 = arith.constant 1310 : index
        %bias_219 = memref.load %in_v[%bidx_219] : memref<1511xi32>
        %smean_219 = memref.load %in_v[%smidx_219] : memref<1511xi32>
        %sscale_219 = memref.load %in_v[%ssidx_219] : memref<1511xi32>
        %dilation_219 = arith.constant 14 : i32
        %padding_219 = arith.constant 56 : i32
        %ki0_219 = arith.constant 2 : i32
        %ki1_219 = arith.constant 3 : i32
        %ki2_219 = arith.constant 6 : i32
        %pad1_219 = arith.constant 1 : i32
        %outidx_219 = arith.constant 219 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_219, %smean_219, %sscale_219, %dilation_219, %padding_219, %ki0_219, %ki1_219, %ki2_219, %pad1_219, %out_v, %outidx_219) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 220: dilation=14 padding=56 kernel=(2,3,7) padding1=0
        %fidx_220 = arith.constant 220 : index
        %bidx_220 = arith.constant 471 : index
        %smidx_220 = arith.constant 891 : index
        %ssidx_220 = arith.constant 1311 : index
        %bias_220 = memref.load %in_v[%bidx_220] : memref<1511xi32>
        %smean_220 = memref.load %in_v[%smidx_220] : memref<1511xi32>
        %sscale_220 = memref.load %in_v[%ssidx_220] : memref<1511xi32>
        %dilation_220 = arith.constant 14 : i32
        %padding_220 = arith.constant 56 : i32
        %ki0_220 = arith.constant 2 : i32
        %ki1_220 = arith.constant 3 : i32
        %ki2_220 = arith.constant 7 : i32
        %pad1_220 = arith.constant 0 : i32
        %outidx_220 = arith.constant 220 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_220, %smean_220, %sscale_220, %dilation_220, %padding_220, %ki0_220, %ki1_220, %ki2_220, %pad1_220, %out_v, %outidx_220) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 221: dilation=14 padding=56 kernel=(2,3,8) padding1=1
        %fidx_221 = arith.constant 221 : index
        %bidx_221 = arith.constant 472 : index
        %smidx_221 = arith.constant 892 : index
        %ssidx_221 = arith.constant 1312 : index
        %bias_221 = memref.load %in_v[%bidx_221] : memref<1511xi32>
        %smean_221 = memref.load %in_v[%smidx_221] : memref<1511xi32>
        %sscale_221 = memref.load %in_v[%ssidx_221] : memref<1511xi32>
        %dilation_221 = arith.constant 14 : i32
        %padding_221 = arith.constant 56 : i32
        %ki0_221 = arith.constant 2 : i32
        %ki1_221 = arith.constant 3 : i32
        %ki2_221 = arith.constant 8 : i32
        %pad1_221 = arith.constant 1 : i32
        %outidx_221 = arith.constant 221 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_221, %smean_221, %sscale_221, %dilation_221, %padding_221, %ki0_221, %ki1_221, %ki2_221, %pad1_221, %out_v, %outidx_221) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 222: dilation=14 padding=56 kernel=(2,4,5) padding1=0
        %fidx_222 = arith.constant 222 : index
        %bidx_222 = arith.constant 473 : index
        %smidx_222 = arith.constant 893 : index
        %ssidx_222 = arith.constant 1313 : index
        %bias_222 = memref.load %in_v[%bidx_222] : memref<1511xi32>
        %smean_222 = memref.load %in_v[%smidx_222] : memref<1511xi32>
        %sscale_222 = memref.load %in_v[%ssidx_222] : memref<1511xi32>
        %dilation_222 = arith.constant 14 : i32
        %padding_222 = arith.constant 56 : i32
        %ki0_222 = arith.constant 2 : i32
        %ki1_222 = arith.constant 4 : i32
        %ki2_222 = arith.constant 5 : i32
        %pad1_222 = arith.constant 0 : i32
        %outidx_222 = arith.constant 222 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_222, %smean_222, %sscale_222, %dilation_222, %padding_222, %ki0_222, %ki1_222, %ki2_222, %pad1_222, %out_v, %outidx_222) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 223: dilation=14 padding=56 kernel=(2,4,6) padding1=1
        %fidx_223 = arith.constant 223 : index
        %bidx_223 = arith.constant 474 : index
        %smidx_223 = arith.constant 894 : index
        %ssidx_223 = arith.constant 1314 : index
        %bias_223 = memref.load %in_v[%bidx_223] : memref<1511xi32>
        %smean_223 = memref.load %in_v[%smidx_223] : memref<1511xi32>
        %sscale_223 = memref.load %in_v[%ssidx_223] : memref<1511xi32>
        %dilation_223 = arith.constant 14 : i32
        %padding_223 = arith.constant 56 : i32
        %ki0_223 = arith.constant 2 : i32
        %ki1_223 = arith.constant 4 : i32
        %ki2_223 = arith.constant 6 : i32
        %pad1_223 = arith.constant 1 : i32
        %outidx_223 = arith.constant 223 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_223, %smean_223, %sscale_223, %dilation_223, %padding_223, %ki0_223, %ki1_223, %ki2_223, %pad1_223, %out_v, %outidx_223) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 224: dilation=14 padding=56 kernel=(2,4,7) padding1=0
        %fidx_224 = arith.constant 224 : index
        %bidx_224 = arith.constant 475 : index
        %smidx_224 = arith.constant 895 : index
        %ssidx_224 = arith.constant 1315 : index
        %bias_224 = memref.load %in_v[%bidx_224] : memref<1511xi32>
        %smean_224 = memref.load %in_v[%smidx_224] : memref<1511xi32>
        %sscale_224 = memref.load %in_v[%ssidx_224] : memref<1511xi32>
        %dilation_224 = arith.constant 14 : i32
        %padding_224 = arith.constant 56 : i32
        %ki0_224 = arith.constant 2 : i32
        %ki1_224 = arith.constant 4 : i32
        %ki2_224 = arith.constant 7 : i32
        %pad1_224 = arith.constant 0 : i32
        %outidx_224 = arith.constant 224 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_224, %smean_224, %sscale_224, %dilation_224, %padding_224, %ki0_224, %ki1_224, %ki2_224, %pad1_224, %out_v, %outidx_224) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 225: dilation=14 padding=56 kernel=(2,4,8) padding1=1
        %fidx_225 = arith.constant 225 : index
        %bidx_225 = arith.constant 476 : index
        %smidx_225 = arith.constant 896 : index
        %ssidx_225 = arith.constant 1316 : index
        %bias_225 = memref.load %in_v[%bidx_225] : memref<1511xi32>
        %smean_225 = memref.load %in_v[%smidx_225] : memref<1511xi32>
        %sscale_225 = memref.load %in_v[%ssidx_225] : memref<1511xi32>
        %dilation_225 = arith.constant 14 : i32
        %padding_225 = arith.constant 56 : i32
        %ki0_225 = arith.constant 2 : i32
        %ki1_225 = arith.constant 4 : i32
        %ki2_225 = arith.constant 8 : i32
        %pad1_225 = arith.constant 1 : i32
        %outidx_225 = arith.constant 225 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_225, %smean_225, %sscale_225, %dilation_225, %padding_225, %ki0_225, %ki1_225, %ki2_225, %pad1_225, %out_v, %outidx_225) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 226: dilation=14 padding=56 kernel=(2,5,6) padding1=0
        %fidx_226 = arith.constant 226 : index
        %bidx_226 = arith.constant 477 : index
        %smidx_226 = arith.constant 897 : index
        %ssidx_226 = arith.constant 1317 : index
        %bias_226 = memref.load %in_v[%bidx_226] : memref<1511xi32>
        %smean_226 = memref.load %in_v[%smidx_226] : memref<1511xi32>
        %sscale_226 = memref.load %in_v[%ssidx_226] : memref<1511xi32>
        %dilation_226 = arith.constant 14 : i32
        %padding_226 = arith.constant 56 : i32
        %ki0_226 = arith.constant 2 : i32
        %ki1_226 = arith.constant 5 : i32
        %ki2_226 = arith.constant 6 : i32
        %pad1_226 = arith.constant 0 : i32
        %outidx_226 = arith.constant 226 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_226, %smean_226, %sscale_226, %dilation_226, %padding_226, %ki0_226, %ki1_226, %ki2_226, %pad1_226, %out_v, %outidx_226) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 227: dilation=14 padding=56 kernel=(2,5,7) padding1=1
        %fidx_227 = arith.constant 227 : index
        %bidx_227 = arith.constant 478 : index
        %smidx_227 = arith.constant 898 : index
        %ssidx_227 = arith.constant 1318 : index
        %bias_227 = memref.load %in_v[%bidx_227] : memref<1511xi32>
        %smean_227 = memref.load %in_v[%smidx_227] : memref<1511xi32>
        %sscale_227 = memref.load %in_v[%ssidx_227] : memref<1511xi32>
        %dilation_227 = arith.constant 14 : i32
        %padding_227 = arith.constant 56 : i32
        %ki0_227 = arith.constant 2 : i32
        %ki1_227 = arith.constant 5 : i32
        %ki2_227 = arith.constant 7 : i32
        %pad1_227 = arith.constant 1 : i32
        %outidx_227 = arith.constant 227 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_227, %smean_227, %sscale_227, %dilation_227, %padding_227, %ki0_227, %ki1_227, %ki2_227, %pad1_227, %out_v, %outidx_227) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 228: dilation=14 padding=56 kernel=(2,5,8) padding1=0
        %fidx_228 = arith.constant 228 : index
        %bidx_228 = arith.constant 479 : index
        %smidx_228 = arith.constant 899 : index
        %ssidx_228 = arith.constant 1319 : index
        %bias_228 = memref.load %in_v[%bidx_228] : memref<1511xi32>
        %smean_228 = memref.load %in_v[%smidx_228] : memref<1511xi32>
        %sscale_228 = memref.load %in_v[%ssidx_228] : memref<1511xi32>
        %dilation_228 = arith.constant 14 : i32
        %padding_228 = arith.constant 56 : i32
        %ki0_228 = arith.constant 2 : i32
        %ki1_228 = arith.constant 5 : i32
        %ki2_228 = arith.constant 8 : i32
        %pad1_228 = arith.constant 0 : i32
        %outidx_228 = arith.constant 228 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_228, %smean_228, %sscale_228, %dilation_228, %padding_228, %ki0_228, %ki1_228, %ki2_228, %pad1_228, %out_v, %outidx_228) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 229: dilation=14 padding=56 kernel=(2,6,7) padding1=1
        %fidx_229 = arith.constant 229 : index
        %bidx_229 = arith.constant 480 : index
        %smidx_229 = arith.constant 900 : index
        %ssidx_229 = arith.constant 1320 : index
        %bias_229 = memref.load %in_v[%bidx_229] : memref<1511xi32>
        %smean_229 = memref.load %in_v[%smidx_229] : memref<1511xi32>
        %sscale_229 = memref.load %in_v[%ssidx_229] : memref<1511xi32>
        %dilation_229 = arith.constant 14 : i32
        %padding_229 = arith.constant 56 : i32
        %ki0_229 = arith.constant 2 : i32
        %ki1_229 = arith.constant 6 : i32
        %ki2_229 = arith.constant 7 : i32
        %pad1_229 = arith.constant 1 : i32
        %outidx_229 = arith.constant 229 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_229, %smean_229, %sscale_229, %dilation_229, %padding_229, %ki0_229, %ki1_229, %ki2_229, %pad1_229, %out_v, %outidx_229) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 230: dilation=14 padding=56 kernel=(2,6,8) padding1=0
        %fidx_230 = arith.constant 230 : index
        %bidx_230 = arith.constant 481 : index
        %smidx_230 = arith.constant 901 : index
        %ssidx_230 = arith.constant 1321 : index
        %bias_230 = memref.load %in_v[%bidx_230] : memref<1511xi32>
        %smean_230 = memref.load %in_v[%smidx_230] : memref<1511xi32>
        %sscale_230 = memref.load %in_v[%ssidx_230] : memref<1511xi32>
        %dilation_230 = arith.constant 14 : i32
        %padding_230 = arith.constant 56 : i32
        %ki0_230 = arith.constant 2 : i32
        %ki1_230 = arith.constant 6 : i32
        %ki2_230 = arith.constant 8 : i32
        %pad1_230 = arith.constant 0 : i32
        %outidx_230 = arith.constant 230 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_230, %smean_230, %sscale_230, %dilation_230, %padding_230, %ki0_230, %ki1_230, %ki2_230, %pad1_230, %out_v, %outidx_230) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 231: dilation=14 padding=56 kernel=(2,7,8) padding1=1
        %fidx_231 = arith.constant 231 : index
        %bidx_231 = arith.constant 482 : index
        %smidx_231 = arith.constant 902 : index
        %ssidx_231 = arith.constant 1322 : index
        %bias_231 = memref.load %in_v[%bidx_231] : memref<1511xi32>
        %smean_231 = memref.load %in_v[%smidx_231] : memref<1511xi32>
        %sscale_231 = memref.load %in_v[%ssidx_231] : memref<1511xi32>
        %dilation_231 = arith.constant 14 : i32
        %padding_231 = arith.constant 56 : i32
        %ki0_231 = arith.constant 2 : i32
        %ki1_231 = arith.constant 7 : i32
        %ki2_231 = arith.constant 8 : i32
        %pad1_231 = arith.constant 1 : i32
        %outidx_231 = arith.constant 231 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_231, %smean_231, %sscale_231, %dilation_231, %padding_231, %ki0_231, %ki1_231, %ki2_231, %pad1_231, %out_v, %outidx_231) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 232: dilation=14 padding=56 kernel=(3,4,5) padding1=0
        %fidx_232 = arith.constant 232 : index
        %bidx_232 = arith.constant 483 : index
        %smidx_232 = arith.constant 903 : index
        %ssidx_232 = arith.constant 1323 : index
        %bias_232 = memref.load %in_v[%bidx_232] : memref<1511xi32>
        %smean_232 = memref.load %in_v[%smidx_232] : memref<1511xi32>
        %sscale_232 = memref.load %in_v[%ssidx_232] : memref<1511xi32>
        %dilation_232 = arith.constant 14 : i32
        %padding_232 = arith.constant 56 : i32
        %ki0_232 = arith.constant 3 : i32
        %ki1_232 = arith.constant 4 : i32
        %ki2_232 = arith.constant 5 : i32
        %pad1_232 = arith.constant 0 : i32
        %outidx_232 = arith.constant 232 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_232, %smean_232, %sscale_232, %dilation_232, %padding_232, %ki0_232, %ki1_232, %ki2_232, %pad1_232, %out_v, %outidx_232) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 233: dilation=14 padding=56 kernel=(3,4,6) padding1=1
        %fidx_233 = arith.constant 233 : index
        %bidx_233 = arith.constant 484 : index
        %smidx_233 = arith.constant 904 : index
        %ssidx_233 = arith.constant 1324 : index
        %bias_233 = memref.load %in_v[%bidx_233] : memref<1511xi32>
        %smean_233 = memref.load %in_v[%smidx_233] : memref<1511xi32>
        %sscale_233 = memref.load %in_v[%ssidx_233] : memref<1511xi32>
        %dilation_233 = arith.constant 14 : i32
        %padding_233 = arith.constant 56 : i32
        %ki0_233 = arith.constant 3 : i32
        %ki1_233 = arith.constant 4 : i32
        %ki2_233 = arith.constant 6 : i32
        %pad1_233 = arith.constant 1 : i32
        %outidx_233 = arith.constant 233 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_233, %smean_233, %sscale_233, %dilation_233, %padding_233, %ki0_233, %ki1_233, %ki2_233, %pad1_233, %out_v, %outidx_233) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 234: dilation=14 padding=56 kernel=(3,4,7) padding1=0
        %fidx_234 = arith.constant 234 : index
        %bidx_234 = arith.constant 485 : index
        %smidx_234 = arith.constant 905 : index
        %ssidx_234 = arith.constant 1325 : index
        %bias_234 = memref.load %in_v[%bidx_234] : memref<1511xi32>
        %smean_234 = memref.load %in_v[%smidx_234] : memref<1511xi32>
        %sscale_234 = memref.load %in_v[%ssidx_234] : memref<1511xi32>
        %dilation_234 = arith.constant 14 : i32
        %padding_234 = arith.constant 56 : i32
        %ki0_234 = arith.constant 3 : i32
        %ki1_234 = arith.constant 4 : i32
        %ki2_234 = arith.constant 7 : i32
        %pad1_234 = arith.constant 0 : i32
        %outidx_234 = arith.constant 234 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_234, %smean_234, %sscale_234, %dilation_234, %padding_234, %ki0_234, %ki1_234, %ki2_234, %pad1_234, %out_v, %outidx_234) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 235: dilation=14 padding=56 kernel=(3,4,8) padding1=1
        %fidx_235 = arith.constant 235 : index
        %bidx_235 = arith.constant 486 : index
        %smidx_235 = arith.constant 906 : index
        %ssidx_235 = arith.constant 1326 : index
        %bias_235 = memref.load %in_v[%bidx_235] : memref<1511xi32>
        %smean_235 = memref.load %in_v[%smidx_235] : memref<1511xi32>
        %sscale_235 = memref.load %in_v[%ssidx_235] : memref<1511xi32>
        %dilation_235 = arith.constant 14 : i32
        %padding_235 = arith.constant 56 : i32
        %ki0_235 = arith.constant 3 : i32
        %ki1_235 = arith.constant 4 : i32
        %ki2_235 = arith.constant 8 : i32
        %pad1_235 = arith.constant 1 : i32
        %outidx_235 = arith.constant 235 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_235, %smean_235, %sscale_235, %dilation_235, %padding_235, %ki0_235, %ki1_235, %ki2_235, %pad1_235, %out_v, %outidx_235) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 236: dilation=14 padding=56 kernel=(3,5,6) padding1=0
        %fidx_236 = arith.constant 236 : index
        %bidx_236 = arith.constant 487 : index
        %smidx_236 = arith.constant 907 : index
        %ssidx_236 = arith.constant 1327 : index
        %bias_236 = memref.load %in_v[%bidx_236] : memref<1511xi32>
        %smean_236 = memref.load %in_v[%smidx_236] : memref<1511xi32>
        %sscale_236 = memref.load %in_v[%ssidx_236] : memref<1511xi32>
        %dilation_236 = arith.constant 14 : i32
        %padding_236 = arith.constant 56 : i32
        %ki0_236 = arith.constant 3 : i32
        %ki1_236 = arith.constant 5 : i32
        %ki2_236 = arith.constant 6 : i32
        %pad1_236 = arith.constant 0 : i32
        %outidx_236 = arith.constant 236 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_236, %smean_236, %sscale_236, %dilation_236, %padding_236, %ki0_236, %ki1_236, %ki2_236, %pad1_236, %out_v, %outidx_236) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 237: dilation=14 padding=56 kernel=(3,5,7) padding1=1
        %fidx_237 = arith.constant 237 : index
        %bidx_237 = arith.constant 488 : index
        %smidx_237 = arith.constant 908 : index
        %ssidx_237 = arith.constant 1328 : index
        %bias_237 = memref.load %in_v[%bidx_237] : memref<1511xi32>
        %smean_237 = memref.load %in_v[%smidx_237] : memref<1511xi32>
        %sscale_237 = memref.load %in_v[%ssidx_237] : memref<1511xi32>
        %dilation_237 = arith.constant 14 : i32
        %padding_237 = arith.constant 56 : i32
        %ki0_237 = arith.constant 3 : i32
        %ki1_237 = arith.constant 5 : i32
        %ki2_237 = arith.constant 7 : i32
        %pad1_237 = arith.constant 1 : i32
        %outidx_237 = arith.constant 237 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_237, %smean_237, %sscale_237, %dilation_237, %padding_237, %ki0_237, %ki1_237, %ki2_237, %pad1_237, %out_v, %outidx_237) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 238: dilation=14 padding=56 kernel=(3,5,8) padding1=0
        %fidx_238 = arith.constant 238 : index
        %bidx_238 = arith.constant 489 : index
        %smidx_238 = arith.constant 909 : index
        %ssidx_238 = arith.constant 1329 : index
        %bias_238 = memref.load %in_v[%bidx_238] : memref<1511xi32>
        %smean_238 = memref.load %in_v[%smidx_238] : memref<1511xi32>
        %sscale_238 = memref.load %in_v[%ssidx_238] : memref<1511xi32>
        %dilation_238 = arith.constant 14 : i32
        %padding_238 = arith.constant 56 : i32
        %ki0_238 = arith.constant 3 : i32
        %ki1_238 = arith.constant 5 : i32
        %ki2_238 = arith.constant 8 : i32
        %pad1_238 = arith.constant 0 : i32
        %outidx_238 = arith.constant 238 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_238, %smean_238, %sscale_238, %dilation_238, %padding_238, %ki0_238, %ki1_238, %ki2_238, %pad1_238, %out_v, %outidx_238) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 239: dilation=14 padding=56 kernel=(3,6,7) padding1=1
        %fidx_239 = arith.constant 239 : index
        %bidx_239 = arith.constant 490 : index
        %smidx_239 = arith.constant 910 : index
        %ssidx_239 = arith.constant 1330 : index
        %bias_239 = memref.load %in_v[%bidx_239] : memref<1511xi32>
        %smean_239 = memref.load %in_v[%smidx_239] : memref<1511xi32>
        %sscale_239 = memref.load %in_v[%ssidx_239] : memref<1511xi32>
        %dilation_239 = arith.constant 14 : i32
        %padding_239 = arith.constant 56 : i32
        %ki0_239 = arith.constant 3 : i32
        %ki1_239 = arith.constant 6 : i32
        %ki2_239 = arith.constant 7 : i32
        %pad1_239 = arith.constant 1 : i32
        %outidx_239 = arith.constant 239 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_239, %smean_239, %sscale_239, %dilation_239, %padding_239, %ki0_239, %ki1_239, %ki2_239, %pad1_239, %out_v, %outidx_239) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 240: dilation=14 padding=56 kernel=(3,6,8) padding1=0
        %fidx_240 = arith.constant 240 : index
        %bidx_240 = arith.constant 491 : index
        %smidx_240 = arith.constant 911 : index
        %ssidx_240 = arith.constant 1331 : index
        %bias_240 = memref.load %in_v[%bidx_240] : memref<1511xi32>
        %smean_240 = memref.load %in_v[%smidx_240] : memref<1511xi32>
        %sscale_240 = memref.load %in_v[%ssidx_240] : memref<1511xi32>
        %dilation_240 = arith.constant 14 : i32
        %padding_240 = arith.constant 56 : i32
        %ki0_240 = arith.constant 3 : i32
        %ki1_240 = arith.constant 6 : i32
        %ki2_240 = arith.constant 8 : i32
        %pad1_240 = arith.constant 0 : i32
        %outidx_240 = arith.constant 240 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_240, %smean_240, %sscale_240, %dilation_240, %padding_240, %ki0_240, %ki1_240, %ki2_240, %pad1_240, %out_v, %outidx_240) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 241: dilation=14 padding=56 kernel=(3,7,8) padding1=1
        %fidx_241 = arith.constant 241 : index
        %bidx_241 = arith.constant 492 : index
        %smidx_241 = arith.constant 912 : index
        %ssidx_241 = arith.constant 1332 : index
        %bias_241 = memref.load %in_v[%bidx_241] : memref<1511xi32>
        %smean_241 = memref.load %in_v[%smidx_241] : memref<1511xi32>
        %sscale_241 = memref.load %in_v[%ssidx_241] : memref<1511xi32>
        %dilation_241 = arith.constant 14 : i32
        %padding_241 = arith.constant 56 : i32
        %ki0_241 = arith.constant 3 : i32
        %ki1_241 = arith.constant 7 : i32
        %ki2_241 = arith.constant 8 : i32
        %pad1_241 = arith.constant 1 : i32
        %outidx_241 = arith.constant 241 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_241, %smean_241, %sscale_241, %dilation_241, %padding_241, %ki0_241, %ki1_241, %ki2_241, %pad1_241, %out_v, %outidx_241) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 242: dilation=14 padding=56 kernel=(4,5,6) padding1=0
        %fidx_242 = arith.constant 242 : index
        %bidx_242 = arith.constant 493 : index
        %smidx_242 = arith.constant 913 : index
        %ssidx_242 = arith.constant 1333 : index
        %bias_242 = memref.load %in_v[%bidx_242] : memref<1511xi32>
        %smean_242 = memref.load %in_v[%smidx_242] : memref<1511xi32>
        %sscale_242 = memref.load %in_v[%ssidx_242] : memref<1511xi32>
        %dilation_242 = arith.constant 14 : i32
        %padding_242 = arith.constant 56 : i32
        %ki0_242 = arith.constant 4 : i32
        %ki1_242 = arith.constant 5 : i32
        %ki2_242 = arith.constant 6 : i32
        %pad1_242 = arith.constant 0 : i32
        %outidx_242 = arith.constant 242 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_242, %smean_242, %sscale_242, %dilation_242, %padding_242, %ki0_242, %ki1_242, %ki2_242, %pad1_242, %out_v, %outidx_242) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 243: dilation=14 padding=56 kernel=(4,5,7) padding1=1
        %fidx_243 = arith.constant 243 : index
        %bidx_243 = arith.constant 494 : index
        %smidx_243 = arith.constant 914 : index
        %ssidx_243 = arith.constant 1334 : index
        %bias_243 = memref.load %in_v[%bidx_243] : memref<1511xi32>
        %smean_243 = memref.load %in_v[%smidx_243] : memref<1511xi32>
        %sscale_243 = memref.load %in_v[%ssidx_243] : memref<1511xi32>
        %dilation_243 = arith.constant 14 : i32
        %padding_243 = arith.constant 56 : i32
        %ki0_243 = arith.constant 4 : i32
        %ki1_243 = arith.constant 5 : i32
        %ki2_243 = arith.constant 7 : i32
        %pad1_243 = arith.constant 1 : i32
        %outidx_243 = arith.constant 243 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_243, %smean_243, %sscale_243, %dilation_243, %padding_243, %ki0_243, %ki1_243, %ki2_243, %pad1_243, %out_v, %outidx_243) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 244: dilation=14 padding=56 kernel=(4,5,8) padding1=0
        %fidx_244 = arith.constant 244 : index
        %bidx_244 = arith.constant 495 : index
        %smidx_244 = arith.constant 915 : index
        %ssidx_244 = arith.constant 1335 : index
        %bias_244 = memref.load %in_v[%bidx_244] : memref<1511xi32>
        %smean_244 = memref.load %in_v[%smidx_244] : memref<1511xi32>
        %sscale_244 = memref.load %in_v[%ssidx_244] : memref<1511xi32>
        %dilation_244 = arith.constant 14 : i32
        %padding_244 = arith.constant 56 : i32
        %ki0_244 = arith.constant 4 : i32
        %ki1_244 = arith.constant 5 : i32
        %ki2_244 = arith.constant 8 : i32
        %pad1_244 = arith.constant 0 : i32
        %outidx_244 = arith.constant 244 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_244, %smean_244, %sscale_244, %dilation_244, %padding_244, %ki0_244, %ki1_244, %ki2_244, %pad1_244, %out_v, %outidx_244) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 245: dilation=14 padding=56 kernel=(4,6,7) padding1=1
        %fidx_245 = arith.constant 245 : index
        %bidx_245 = arith.constant 496 : index
        %smidx_245 = arith.constant 916 : index
        %ssidx_245 = arith.constant 1336 : index
        %bias_245 = memref.load %in_v[%bidx_245] : memref<1511xi32>
        %smean_245 = memref.load %in_v[%smidx_245] : memref<1511xi32>
        %sscale_245 = memref.load %in_v[%ssidx_245] : memref<1511xi32>
        %dilation_245 = arith.constant 14 : i32
        %padding_245 = arith.constant 56 : i32
        %ki0_245 = arith.constant 4 : i32
        %ki1_245 = arith.constant 6 : i32
        %ki2_245 = arith.constant 7 : i32
        %pad1_245 = arith.constant 1 : i32
        %outidx_245 = arith.constant 245 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_245, %smean_245, %sscale_245, %dilation_245, %padding_245, %ki0_245, %ki1_245, %ki2_245, %pad1_245, %out_v, %outidx_245) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 246: dilation=14 padding=56 kernel=(4,6,8) padding1=0
        %fidx_246 = arith.constant 246 : index
        %bidx_246 = arith.constant 497 : index
        %smidx_246 = arith.constant 917 : index
        %ssidx_246 = arith.constant 1337 : index
        %bias_246 = memref.load %in_v[%bidx_246] : memref<1511xi32>
        %smean_246 = memref.load %in_v[%smidx_246] : memref<1511xi32>
        %sscale_246 = memref.load %in_v[%ssidx_246] : memref<1511xi32>
        %dilation_246 = arith.constant 14 : i32
        %padding_246 = arith.constant 56 : i32
        %ki0_246 = arith.constant 4 : i32
        %ki1_246 = arith.constant 6 : i32
        %ki2_246 = arith.constant 8 : i32
        %pad1_246 = arith.constant 0 : i32
        %outidx_246 = arith.constant 246 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_246, %smean_246, %sscale_246, %dilation_246, %padding_246, %ki0_246, %ki1_246, %ki2_246, %pad1_246, %out_v, %outidx_246) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 247: dilation=14 padding=56 kernel=(4,7,8) padding1=1
        %fidx_247 = arith.constant 247 : index
        %bidx_247 = arith.constant 498 : index
        %smidx_247 = arith.constant 918 : index
        %ssidx_247 = arith.constant 1338 : index
        %bias_247 = memref.load %in_v[%bidx_247] : memref<1511xi32>
        %smean_247 = memref.load %in_v[%smidx_247] : memref<1511xi32>
        %sscale_247 = memref.load %in_v[%ssidx_247] : memref<1511xi32>
        %dilation_247 = arith.constant 14 : i32
        %padding_247 = arith.constant 56 : i32
        %ki0_247 = arith.constant 4 : i32
        %ki1_247 = arith.constant 7 : i32
        %ki2_247 = arith.constant 8 : i32
        %pad1_247 = arith.constant 1 : i32
        %outidx_247 = arith.constant 247 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_247, %smean_247, %sscale_247, %dilation_247, %padding_247, %ki0_247, %ki1_247, %ki2_247, %pad1_247, %out_v, %outidx_247) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 248: dilation=14 padding=56 kernel=(5,6,7) padding1=0
        %fidx_248 = arith.constant 248 : index
        %bidx_248 = arith.constant 499 : index
        %smidx_248 = arith.constant 919 : index
        %ssidx_248 = arith.constant 1339 : index
        %bias_248 = memref.load %in_v[%bidx_248] : memref<1511xi32>
        %smean_248 = memref.load %in_v[%smidx_248] : memref<1511xi32>
        %sscale_248 = memref.load %in_v[%ssidx_248] : memref<1511xi32>
        %dilation_248 = arith.constant 14 : i32
        %padding_248 = arith.constant 56 : i32
        %ki0_248 = arith.constant 5 : i32
        %ki1_248 = arith.constant 6 : i32
        %ki2_248 = arith.constant 7 : i32
        %pad1_248 = arith.constant 0 : i32
        %outidx_248 = arith.constant 248 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_248, %smean_248, %sscale_248, %dilation_248, %padding_248, %ki0_248, %ki1_248, %ki2_248, %pad1_248, %out_v, %outidx_248) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 249: dilation=14 padding=56 kernel=(5,6,8) padding1=1
        %fidx_249 = arith.constant 249 : index
        %bidx_249 = arith.constant 500 : index
        %smidx_249 = arith.constant 920 : index
        %ssidx_249 = arith.constant 1340 : index
        %bias_249 = memref.load %in_v[%bidx_249] : memref<1511xi32>
        %smean_249 = memref.load %in_v[%smidx_249] : memref<1511xi32>
        %sscale_249 = memref.load %in_v[%ssidx_249] : memref<1511xi32>
        %dilation_249 = arith.constant 14 : i32
        %padding_249 = arith.constant 56 : i32
        %ki0_249 = arith.constant 5 : i32
        %ki1_249 = arith.constant 6 : i32
        %ki2_249 = arith.constant 8 : i32
        %pad1_249 = arith.constant 1 : i32
        %outidx_249 = arith.constant 249 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_249, %smean_249, %sscale_249, %dilation_249, %padding_249, %ki0_249, %ki1_249, %ki2_249, %pad1_249, %out_v, %outidx_249) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 250: dilation=14 padding=56 kernel=(5,7,8) padding1=0
        %fidx_250 = arith.constant 250 : index
        %bidx_250 = arith.constant 501 : index
        %smidx_250 = arith.constant 921 : index
        %ssidx_250 = arith.constant 1341 : index
        %bias_250 = memref.load %in_v[%bidx_250] : memref<1511xi32>
        %smean_250 = memref.load %in_v[%smidx_250] : memref<1511xi32>
        %sscale_250 = memref.load %in_v[%ssidx_250] : memref<1511xi32>
        %dilation_250 = arith.constant 14 : i32
        %padding_250 = arith.constant 56 : i32
        %ki0_250 = arith.constant 5 : i32
        %ki1_250 = arith.constant 7 : i32
        %ki2_250 = arith.constant 8 : i32
        %pad1_250 = arith.constant 0 : i32
        %outidx_250 = arith.constant 250 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_250, %smean_250, %sscale_250, %dilation_250, %padding_250, %ki0_250, %ki1_250, %ki2_250, %pad1_250, %out_v, %outidx_250) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 251: dilation=14 padding=56 kernel=(6,7,8) padding1=1
        %fidx_251 = arith.constant 251 : index
        %bidx_251 = arith.constant 502 : index
        %smidx_251 = arith.constant 922 : index
        %ssidx_251 = arith.constant 1342 : index
        %bias_251 = memref.load %in_v[%bidx_251] : memref<1511xi32>
        %smean_251 = memref.load %in_v[%smidx_251] : memref<1511xi32>
        %sscale_251 = memref.load %in_v[%ssidx_251] : memref<1511xi32>
        %dilation_251 = arith.constant 14 : i32
        %padding_251 = arith.constant 56 : i32
        %ki0_251 = arith.constant 6 : i32
        %ki1_251 = arith.constant 7 : i32
        %ki2_251 = arith.constant 8 : i32
        %pad1_251 = arith.constant 1 : i32
        %outidx_251 = arith.constant 251 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_251, %smean_251, %sscale_251, %dilation_251, %padding_251, %ki0_251, %ki1_251, %ki2_251, %pad1_251, %out_v, %outidx_251) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 252: dilation=21 padding=84 kernel=(0,1,2) padding1=1
        %fidx_252 = arith.constant 252 : index
        %bidx_252 = arith.constant 503 : index
        %smidx_252 = arith.constant 923 : index
        %ssidx_252 = arith.constant 1343 : index
        %bias_252 = memref.load %in_v[%bidx_252] : memref<1511xi32>
        %smean_252 = memref.load %in_v[%smidx_252] : memref<1511xi32>
        %sscale_252 = memref.load %in_v[%ssidx_252] : memref<1511xi32>
        %dilation_252 = arith.constant 21 : i32
        %padding_252 = arith.constant 84 : i32
        %ki0_252 = arith.constant 0 : i32
        %ki1_252 = arith.constant 1 : i32
        %ki2_252 = arith.constant 2 : i32
        %pad1_252 = arith.constant 1 : i32
        %outidx_252 = arith.constant 252 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_252, %smean_252, %sscale_252, %dilation_252, %padding_252, %ki0_252, %ki1_252, %ki2_252, %pad1_252, %out_v, %outidx_252) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 253: dilation=21 padding=84 kernel=(0,1,3) padding1=0
        %fidx_253 = arith.constant 253 : index
        %bidx_253 = arith.constant 504 : index
        %smidx_253 = arith.constant 924 : index
        %ssidx_253 = arith.constant 1344 : index
        %bias_253 = memref.load %in_v[%bidx_253] : memref<1511xi32>
        %smean_253 = memref.load %in_v[%smidx_253] : memref<1511xi32>
        %sscale_253 = memref.load %in_v[%ssidx_253] : memref<1511xi32>
        %dilation_253 = arith.constant 21 : i32
        %padding_253 = arith.constant 84 : i32
        %ki0_253 = arith.constant 0 : i32
        %ki1_253 = arith.constant 1 : i32
        %ki2_253 = arith.constant 3 : i32
        %pad1_253 = arith.constant 0 : i32
        %outidx_253 = arith.constant 253 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_253, %smean_253, %sscale_253, %dilation_253, %padding_253, %ki0_253, %ki1_253, %ki2_253, %pad1_253, %out_v, %outidx_253) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 254: dilation=21 padding=84 kernel=(0,1,4) padding1=1
        %fidx_254 = arith.constant 254 : index
        %bidx_254 = arith.constant 505 : index
        %smidx_254 = arith.constant 925 : index
        %ssidx_254 = arith.constant 1345 : index
        %bias_254 = memref.load %in_v[%bidx_254] : memref<1511xi32>
        %smean_254 = memref.load %in_v[%smidx_254] : memref<1511xi32>
        %sscale_254 = memref.load %in_v[%ssidx_254] : memref<1511xi32>
        %dilation_254 = arith.constant 21 : i32
        %padding_254 = arith.constant 84 : i32
        %ki0_254 = arith.constant 0 : i32
        %ki1_254 = arith.constant 1 : i32
        %ki2_254 = arith.constant 4 : i32
        %pad1_254 = arith.constant 1 : i32
        %outidx_254 = arith.constant 254 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_254, %smean_254, %sscale_254, %dilation_254, %padding_254, %ki0_254, %ki1_254, %ki2_254, %pad1_254, %out_v, %outidx_254) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 255: dilation=21 padding=84 kernel=(0,1,5) padding1=0
        %fidx_255 = arith.constant 255 : index
        %bidx_255 = arith.constant 506 : index
        %smidx_255 = arith.constant 926 : index
        %ssidx_255 = arith.constant 1346 : index
        %bias_255 = memref.load %in_v[%bidx_255] : memref<1511xi32>
        %smean_255 = memref.load %in_v[%smidx_255] : memref<1511xi32>
        %sscale_255 = memref.load %in_v[%ssidx_255] : memref<1511xi32>
        %dilation_255 = arith.constant 21 : i32
        %padding_255 = arith.constant 84 : i32
        %ki0_255 = arith.constant 0 : i32
        %ki1_255 = arith.constant 1 : i32
        %ki2_255 = arith.constant 5 : i32
        %pad1_255 = arith.constant 0 : i32
        %outidx_255 = arith.constant 255 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_255, %smean_255, %sscale_255, %dilation_255, %padding_255, %ki0_255, %ki1_255, %ki2_255, %pad1_255, %out_v, %outidx_255) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 256: dilation=21 padding=84 kernel=(0,1,6) padding1=1
        %fidx_256 = arith.constant 256 : index
        %bidx_256 = arith.constant 507 : index
        %smidx_256 = arith.constant 927 : index
        %ssidx_256 = arith.constant 1347 : index
        %bias_256 = memref.load %in_v[%bidx_256] : memref<1511xi32>
        %smean_256 = memref.load %in_v[%smidx_256] : memref<1511xi32>
        %sscale_256 = memref.load %in_v[%ssidx_256] : memref<1511xi32>
        %dilation_256 = arith.constant 21 : i32
        %padding_256 = arith.constant 84 : i32
        %ki0_256 = arith.constant 0 : i32
        %ki1_256 = arith.constant 1 : i32
        %ki2_256 = arith.constant 6 : i32
        %pad1_256 = arith.constant 1 : i32
        %outidx_256 = arith.constant 256 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_256, %smean_256, %sscale_256, %dilation_256, %padding_256, %ki0_256, %ki1_256, %ki2_256, %pad1_256, %out_v, %outidx_256) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 257: dilation=21 padding=84 kernel=(0,1,7) padding1=0
        %fidx_257 = arith.constant 257 : index
        %bidx_257 = arith.constant 508 : index
        %smidx_257 = arith.constant 928 : index
        %ssidx_257 = arith.constant 1348 : index
        %bias_257 = memref.load %in_v[%bidx_257] : memref<1511xi32>
        %smean_257 = memref.load %in_v[%smidx_257] : memref<1511xi32>
        %sscale_257 = memref.load %in_v[%ssidx_257] : memref<1511xi32>
        %dilation_257 = arith.constant 21 : i32
        %padding_257 = arith.constant 84 : i32
        %ki0_257 = arith.constant 0 : i32
        %ki1_257 = arith.constant 1 : i32
        %ki2_257 = arith.constant 7 : i32
        %pad1_257 = arith.constant 0 : i32
        %outidx_257 = arith.constant 257 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_257, %smean_257, %sscale_257, %dilation_257, %padding_257, %ki0_257, %ki1_257, %ki2_257, %pad1_257, %out_v, %outidx_257) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 258: dilation=21 padding=84 kernel=(0,1,8) padding1=1
        %fidx_258 = arith.constant 258 : index
        %bidx_258 = arith.constant 509 : index
        %smidx_258 = arith.constant 929 : index
        %ssidx_258 = arith.constant 1349 : index
        %bias_258 = memref.load %in_v[%bidx_258] : memref<1511xi32>
        %smean_258 = memref.load %in_v[%smidx_258] : memref<1511xi32>
        %sscale_258 = memref.load %in_v[%ssidx_258] : memref<1511xi32>
        %dilation_258 = arith.constant 21 : i32
        %padding_258 = arith.constant 84 : i32
        %ki0_258 = arith.constant 0 : i32
        %ki1_258 = arith.constant 1 : i32
        %ki2_258 = arith.constant 8 : i32
        %pad1_258 = arith.constant 1 : i32
        %outidx_258 = arith.constant 258 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_258, %smean_258, %sscale_258, %dilation_258, %padding_258, %ki0_258, %ki1_258, %ki2_258, %pad1_258, %out_v, %outidx_258) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 259: dilation=21 padding=84 kernel=(0,2,3) padding1=0
        %fidx_259 = arith.constant 259 : index
        %bidx_259 = arith.constant 510 : index
        %smidx_259 = arith.constant 930 : index
        %ssidx_259 = arith.constant 1350 : index
        %bias_259 = memref.load %in_v[%bidx_259] : memref<1511xi32>
        %smean_259 = memref.load %in_v[%smidx_259] : memref<1511xi32>
        %sscale_259 = memref.load %in_v[%ssidx_259] : memref<1511xi32>
        %dilation_259 = arith.constant 21 : i32
        %padding_259 = arith.constant 84 : i32
        %ki0_259 = arith.constant 0 : i32
        %ki1_259 = arith.constant 2 : i32
        %ki2_259 = arith.constant 3 : i32
        %pad1_259 = arith.constant 0 : i32
        %outidx_259 = arith.constant 259 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_259, %smean_259, %sscale_259, %dilation_259, %padding_259, %ki0_259, %ki1_259, %ki2_259, %pad1_259, %out_v, %outidx_259) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 260: dilation=21 padding=84 kernel=(0,2,4) padding1=1
        %fidx_260 = arith.constant 260 : index
        %bidx_260 = arith.constant 511 : index
        %smidx_260 = arith.constant 931 : index
        %ssidx_260 = arith.constant 1351 : index
        %bias_260 = memref.load %in_v[%bidx_260] : memref<1511xi32>
        %smean_260 = memref.load %in_v[%smidx_260] : memref<1511xi32>
        %sscale_260 = memref.load %in_v[%ssidx_260] : memref<1511xi32>
        %dilation_260 = arith.constant 21 : i32
        %padding_260 = arith.constant 84 : i32
        %ki0_260 = arith.constant 0 : i32
        %ki1_260 = arith.constant 2 : i32
        %ki2_260 = arith.constant 4 : i32
        %pad1_260 = arith.constant 1 : i32
        %outidx_260 = arith.constant 260 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_260, %smean_260, %sscale_260, %dilation_260, %padding_260, %ki0_260, %ki1_260, %ki2_260, %pad1_260, %out_v, %outidx_260) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 261: dilation=21 padding=84 kernel=(0,2,5) padding1=0
        %fidx_261 = arith.constant 261 : index
        %bidx_261 = arith.constant 512 : index
        %smidx_261 = arith.constant 932 : index
        %ssidx_261 = arith.constant 1352 : index
        %bias_261 = memref.load %in_v[%bidx_261] : memref<1511xi32>
        %smean_261 = memref.load %in_v[%smidx_261] : memref<1511xi32>
        %sscale_261 = memref.load %in_v[%ssidx_261] : memref<1511xi32>
        %dilation_261 = arith.constant 21 : i32
        %padding_261 = arith.constant 84 : i32
        %ki0_261 = arith.constant 0 : i32
        %ki1_261 = arith.constant 2 : i32
        %ki2_261 = arith.constant 5 : i32
        %pad1_261 = arith.constant 0 : i32
        %outidx_261 = arith.constant 261 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_261, %smean_261, %sscale_261, %dilation_261, %padding_261, %ki0_261, %ki1_261, %ki2_261, %pad1_261, %out_v, %outidx_261) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 262: dilation=21 padding=84 kernel=(0,2,6) padding1=1
        %fidx_262 = arith.constant 262 : index
        %bidx_262 = arith.constant 513 : index
        %smidx_262 = arith.constant 933 : index
        %ssidx_262 = arith.constant 1353 : index
        %bias_262 = memref.load %in_v[%bidx_262] : memref<1511xi32>
        %smean_262 = memref.load %in_v[%smidx_262] : memref<1511xi32>
        %sscale_262 = memref.load %in_v[%ssidx_262] : memref<1511xi32>
        %dilation_262 = arith.constant 21 : i32
        %padding_262 = arith.constant 84 : i32
        %ki0_262 = arith.constant 0 : i32
        %ki1_262 = arith.constant 2 : i32
        %ki2_262 = arith.constant 6 : i32
        %pad1_262 = arith.constant 1 : i32
        %outidx_262 = arith.constant 262 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_262, %smean_262, %sscale_262, %dilation_262, %padding_262, %ki0_262, %ki1_262, %ki2_262, %pad1_262, %out_v, %outidx_262) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 263: dilation=21 padding=84 kernel=(0,2,7) padding1=0
        %fidx_263 = arith.constant 263 : index
        %bidx_263 = arith.constant 514 : index
        %smidx_263 = arith.constant 934 : index
        %ssidx_263 = arith.constant 1354 : index
        %bias_263 = memref.load %in_v[%bidx_263] : memref<1511xi32>
        %smean_263 = memref.load %in_v[%smidx_263] : memref<1511xi32>
        %sscale_263 = memref.load %in_v[%ssidx_263] : memref<1511xi32>
        %dilation_263 = arith.constant 21 : i32
        %padding_263 = arith.constant 84 : i32
        %ki0_263 = arith.constant 0 : i32
        %ki1_263 = arith.constant 2 : i32
        %ki2_263 = arith.constant 7 : i32
        %pad1_263 = arith.constant 0 : i32
        %outidx_263 = arith.constant 263 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_263, %smean_263, %sscale_263, %dilation_263, %padding_263, %ki0_263, %ki1_263, %ki2_263, %pad1_263, %out_v, %outidx_263) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 264: dilation=21 padding=84 kernel=(0,2,8) padding1=1
        %fidx_264 = arith.constant 264 : index
        %bidx_264 = arith.constant 515 : index
        %smidx_264 = arith.constant 935 : index
        %ssidx_264 = arith.constant 1355 : index
        %bias_264 = memref.load %in_v[%bidx_264] : memref<1511xi32>
        %smean_264 = memref.load %in_v[%smidx_264] : memref<1511xi32>
        %sscale_264 = memref.load %in_v[%ssidx_264] : memref<1511xi32>
        %dilation_264 = arith.constant 21 : i32
        %padding_264 = arith.constant 84 : i32
        %ki0_264 = arith.constant 0 : i32
        %ki1_264 = arith.constant 2 : i32
        %ki2_264 = arith.constant 8 : i32
        %pad1_264 = arith.constant 1 : i32
        %outidx_264 = arith.constant 264 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_264, %smean_264, %sscale_264, %dilation_264, %padding_264, %ki0_264, %ki1_264, %ki2_264, %pad1_264, %out_v, %outidx_264) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 265: dilation=21 padding=84 kernel=(0,3,4) padding1=0
        %fidx_265 = arith.constant 265 : index
        %bidx_265 = arith.constant 516 : index
        %smidx_265 = arith.constant 936 : index
        %ssidx_265 = arith.constant 1356 : index
        %bias_265 = memref.load %in_v[%bidx_265] : memref<1511xi32>
        %smean_265 = memref.load %in_v[%smidx_265] : memref<1511xi32>
        %sscale_265 = memref.load %in_v[%ssidx_265] : memref<1511xi32>
        %dilation_265 = arith.constant 21 : i32
        %padding_265 = arith.constant 84 : i32
        %ki0_265 = arith.constant 0 : i32
        %ki1_265 = arith.constant 3 : i32
        %ki2_265 = arith.constant 4 : i32
        %pad1_265 = arith.constant 0 : i32
        %outidx_265 = arith.constant 265 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_265, %smean_265, %sscale_265, %dilation_265, %padding_265, %ki0_265, %ki1_265, %ki2_265, %pad1_265, %out_v, %outidx_265) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 266: dilation=21 padding=84 kernel=(0,3,5) padding1=1
        %fidx_266 = arith.constant 266 : index
        %bidx_266 = arith.constant 517 : index
        %smidx_266 = arith.constant 937 : index
        %ssidx_266 = arith.constant 1357 : index
        %bias_266 = memref.load %in_v[%bidx_266] : memref<1511xi32>
        %smean_266 = memref.load %in_v[%smidx_266] : memref<1511xi32>
        %sscale_266 = memref.load %in_v[%ssidx_266] : memref<1511xi32>
        %dilation_266 = arith.constant 21 : i32
        %padding_266 = arith.constant 84 : i32
        %ki0_266 = arith.constant 0 : i32
        %ki1_266 = arith.constant 3 : i32
        %ki2_266 = arith.constant 5 : i32
        %pad1_266 = arith.constant 1 : i32
        %outidx_266 = arith.constant 266 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_266, %smean_266, %sscale_266, %dilation_266, %padding_266, %ki0_266, %ki1_266, %ki2_266, %pad1_266, %out_v, %outidx_266) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 267: dilation=21 padding=84 kernel=(0,3,6) padding1=0
        %fidx_267 = arith.constant 267 : index
        %bidx_267 = arith.constant 518 : index
        %smidx_267 = arith.constant 938 : index
        %ssidx_267 = arith.constant 1358 : index
        %bias_267 = memref.load %in_v[%bidx_267] : memref<1511xi32>
        %smean_267 = memref.load %in_v[%smidx_267] : memref<1511xi32>
        %sscale_267 = memref.load %in_v[%ssidx_267] : memref<1511xi32>
        %dilation_267 = arith.constant 21 : i32
        %padding_267 = arith.constant 84 : i32
        %ki0_267 = arith.constant 0 : i32
        %ki1_267 = arith.constant 3 : i32
        %ki2_267 = arith.constant 6 : i32
        %pad1_267 = arith.constant 0 : i32
        %outidx_267 = arith.constant 267 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_267, %smean_267, %sscale_267, %dilation_267, %padding_267, %ki0_267, %ki1_267, %ki2_267, %pad1_267, %out_v, %outidx_267) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 268: dilation=21 padding=84 kernel=(0,3,7) padding1=1
        %fidx_268 = arith.constant 268 : index
        %bidx_268 = arith.constant 519 : index
        %smidx_268 = arith.constant 939 : index
        %ssidx_268 = arith.constant 1359 : index
        %bias_268 = memref.load %in_v[%bidx_268] : memref<1511xi32>
        %smean_268 = memref.load %in_v[%smidx_268] : memref<1511xi32>
        %sscale_268 = memref.load %in_v[%ssidx_268] : memref<1511xi32>
        %dilation_268 = arith.constant 21 : i32
        %padding_268 = arith.constant 84 : i32
        %ki0_268 = arith.constant 0 : i32
        %ki1_268 = arith.constant 3 : i32
        %ki2_268 = arith.constant 7 : i32
        %pad1_268 = arith.constant 1 : i32
        %outidx_268 = arith.constant 268 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_268, %smean_268, %sscale_268, %dilation_268, %padding_268, %ki0_268, %ki1_268, %ki2_268, %pad1_268, %out_v, %outidx_268) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 269: dilation=21 padding=84 kernel=(0,3,8) padding1=0
        %fidx_269 = arith.constant 269 : index
        %bidx_269 = arith.constant 520 : index
        %smidx_269 = arith.constant 940 : index
        %ssidx_269 = arith.constant 1360 : index
        %bias_269 = memref.load %in_v[%bidx_269] : memref<1511xi32>
        %smean_269 = memref.load %in_v[%smidx_269] : memref<1511xi32>
        %sscale_269 = memref.load %in_v[%ssidx_269] : memref<1511xi32>
        %dilation_269 = arith.constant 21 : i32
        %padding_269 = arith.constant 84 : i32
        %ki0_269 = arith.constant 0 : i32
        %ki1_269 = arith.constant 3 : i32
        %ki2_269 = arith.constant 8 : i32
        %pad1_269 = arith.constant 0 : i32
        %outidx_269 = arith.constant 269 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_269, %smean_269, %sscale_269, %dilation_269, %padding_269, %ki0_269, %ki1_269, %ki2_269, %pad1_269, %out_v, %outidx_269) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 270: dilation=21 padding=84 kernel=(0,4,5) padding1=1
        %fidx_270 = arith.constant 270 : index
        %bidx_270 = arith.constant 521 : index
        %smidx_270 = arith.constant 941 : index
        %ssidx_270 = arith.constant 1361 : index
        %bias_270 = memref.load %in_v[%bidx_270] : memref<1511xi32>
        %smean_270 = memref.load %in_v[%smidx_270] : memref<1511xi32>
        %sscale_270 = memref.load %in_v[%ssidx_270] : memref<1511xi32>
        %dilation_270 = arith.constant 21 : i32
        %padding_270 = arith.constant 84 : i32
        %ki0_270 = arith.constant 0 : i32
        %ki1_270 = arith.constant 4 : i32
        %ki2_270 = arith.constant 5 : i32
        %pad1_270 = arith.constant 1 : i32
        %outidx_270 = arith.constant 270 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_270, %smean_270, %sscale_270, %dilation_270, %padding_270, %ki0_270, %ki1_270, %ki2_270, %pad1_270, %out_v, %outidx_270) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 271: dilation=21 padding=84 kernel=(0,4,6) padding1=0
        %fidx_271 = arith.constant 271 : index
        %bidx_271 = arith.constant 522 : index
        %smidx_271 = arith.constant 942 : index
        %ssidx_271 = arith.constant 1362 : index
        %bias_271 = memref.load %in_v[%bidx_271] : memref<1511xi32>
        %smean_271 = memref.load %in_v[%smidx_271] : memref<1511xi32>
        %sscale_271 = memref.load %in_v[%ssidx_271] : memref<1511xi32>
        %dilation_271 = arith.constant 21 : i32
        %padding_271 = arith.constant 84 : i32
        %ki0_271 = arith.constant 0 : i32
        %ki1_271 = arith.constant 4 : i32
        %ki2_271 = arith.constant 6 : i32
        %pad1_271 = arith.constant 0 : i32
        %outidx_271 = arith.constant 271 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_271, %smean_271, %sscale_271, %dilation_271, %padding_271, %ki0_271, %ki1_271, %ki2_271, %pad1_271, %out_v, %outidx_271) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 272: dilation=21 padding=84 kernel=(0,4,7) padding1=1
        %fidx_272 = arith.constant 272 : index
        %bidx_272 = arith.constant 523 : index
        %smidx_272 = arith.constant 943 : index
        %ssidx_272 = arith.constant 1363 : index
        %bias_272 = memref.load %in_v[%bidx_272] : memref<1511xi32>
        %smean_272 = memref.load %in_v[%smidx_272] : memref<1511xi32>
        %sscale_272 = memref.load %in_v[%ssidx_272] : memref<1511xi32>
        %dilation_272 = arith.constant 21 : i32
        %padding_272 = arith.constant 84 : i32
        %ki0_272 = arith.constant 0 : i32
        %ki1_272 = arith.constant 4 : i32
        %ki2_272 = arith.constant 7 : i32
        %pad1_272 = arith.constant 1 : i32
        %outidx_272 = arith.constant 272 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_272, %smean_272, %sscale_272, %dilation_272, %padding_272, %ki0_272, %ki1_272, %ki2_272, %pad1_272, %out_v, %outidx_272) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 273: dilation=21 padding=84 kernel=(0,4,8) padding1=0
        %fidx_273 = arith.constant 273 : index
        %bidx_273 = arith.constant 524 : index
        %smidx_273 = arith.constant 944 : index
        %ssidx_273 = arith.constant 1364 : index
        %bias_273 = memref.load %in_v[%bidx_273] : memref<1511xi32>
        %smean_273 = memref.load %in_v[%smidx_273] : memref<1511xi32>
        %sscale_273 = memref.load %in_v[%ssidx_273] : memref<1511xi32>
        %dilation_273 = arith.constant 21 : i32
        %padding_273 = arith.constant 84 : i32
        %ki0_273 = arith.constant 0 : i32
        %ki1_273 = arith.constant 4 : i32
        %ki2_273 = arith.constant 8 : i32
        %pad1_273 = arith.constant 0 : i32
        %outidx_273 = arith.constant 273 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_273, %smean_273, %sscale_273, %dilation_273, %padding_273, %ki0_273, %ki1_273, %ki2_273, %pad1_273, %out_v, %outidx_273) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 274: dilation=21 padding=84 kernel=(0,5,6) padding1=1
        %fidx_274 = arith.constant 274 : index
        %bidx_274 = arith.constant 525 : index
        %smidx_274 = arith.constant 945 : index
        %ssidx_274 = arith.constant 1365 : index
        %bias_274 = memref.load %in_v[%bidx_274] : memref<1511xi32>
        %smean_274 = memref.load %in_v[%smidx_274] : memref<1511xi32>
        %sscale_274 = memref.load %in_v[%ssidx_274] : memref<1511xi32>
        %dilation_274 = arith.constant 21 : i32
        %padding_274 = arith.constant 84 : i32
        %ki0_274 = arith.constant 0 : i32
        %ki1_274 = arith.constant 5 : i32
        %ki2_274 = arith.constant 6 : i32
        %pad1_274 = arith.constant 1 : i32
        %outidx_274 = arith.constant 274 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_274, %smean_274, %sscale_274, %dilation_274, %padding_274, %ki0_274, %ki1_274, %ki2_274, %pad1_274, %out_v, %outidx_274) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 275: dilation=21 padding=84 kernel=(0,5,7) padding1=0
        %fidx_275 = arith.constant 275 : index
        %bidx_275 = arith.constant 526 : index
        %smidx_275 = arith.constant 946 : index
        %ssidx_275 = arith.constant 1366 : index
        %bias_275 = memref.load %in_v[%bidx_275] : memref<1511xi32>
        %smean_275 = memref.load %in_v[%smidx_275] : memref<1511xi32>
        %sscale_275 = memref.load %in_v[%ssidx_275] : memref<1511xi32>
        %dilation_275 = arith.constant 21 : i32
        %padding_275 = arith.constant 84 : i32
        %ki0_275 = arith.constant 0 : i32
        %ki1_275 = arith.constant 5 : i32
        %ki2_275 = arith.constant 7 : i32
        %pad1_275 = arith.constant 0 : i32
        %outidx_275 = arith.constant 275 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_275, %smean_275, %sscale_275, %dilation_275, %padding_275, %ki0_275, %ki1_275, %ki2_275, %pad1_275, %out_v, %outidx_275) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 276: dilation=21 padding=84 kernel=(0,5,8) padding1=1
        %fidx_276 = arith.constant 276 : index
        %bidx_276 = arith.constant 527 : index
        %smidx_276 = arith.constant 947 : index
        %ssidx_276 = arith.constant 1367 : index
        %bias_276 = memref.load %in_v[%bidx_276] : memref<1511xi32>
        %smean_276 = memref.load %in_v[%smidx_276] : memref<1511xi32>
        %sscale_276 = memref.load %in_v[%ssidx_276] : memref<1511xi32>
        %dilation_276 = arith.constant 21 : i32
        %padding_276 = arith.constant 84 : i32
        %ki0_276 = arith.constant 0 : i32
        %ki1_276 = arith.constant 5 : i32
        %ki2_276 = arith.constant 8 : i32
        %pad1_276 = arith.constant 1 : i32
        %outidx_276 = arith.constant 276 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_276, %smean_276, %sscale_276, %dilation_276, %padding_276, %ki0_276, %ki1_276, %ki2_276, %pad1_276, %out_v, %outidx_276) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 277: dilation=21 padding=84 kernel=(0,6,7) padding1=0
        %fidx_277 = arith.constant 277 : index
        %bidx_277 = arith.constant 528 : index
        %smidx_277 = arith.constant 948 : index
        %ssidx_277 = arith.constant 1368 : index
        %bias_277 = memref.load %in_v[%bidx_277] : memref<1511xi32>
        %smean_277 = memref.load %in_v[%smidx_277] : memref<1511xi32>
        %sscale_277 = memref.load %in_v[%ssidx_277] : memref<1511xi32>
        %dilation_277 = arith.constant 21 : i32
        %padding_277 = arith.constant 84 : i32
        %ki0_277 = arith.constant 0 : i32
        %ki1_277 = arith.constant 6 : i32
        %ki2_277 = arith.constant 7 : i32
        %pad1_277 = arith.constant 0 : i32
        %outidx_277 = arith.constant 277 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_277, %smean_277, %sscale_277, %dilation_277, %padding_277, %ki0_277, %ki1_277, %ki2_277, %pad1_277, %out_v, %outidx_277) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 278: dilation=21 padding=84 kernel=(0,6,8) padding1=1
        %fidx_278 = arith.constant 278 : index
        %bidx_278 = arith.constant 529 : index
        %smidx_278 = arith.constant 949 : index
        %ssidx_278 = arith.constant 1369 : index
        %bias_278 = memref.load %in_v[%bidx_278] : memref<1511xi32>
        %smean_278 = memref.load %in_v[%smidx_278] : memref<1511xi32>
        %sscale_278 = memref.load %in_v[%ssidx_278] : memref<1511xi32>
        %dilation_278 = arith.constant 21 : i32
        %padding_278 = arith.constant 84 : i32
        %ki0_278 = arith.constant 0 : i32
        %ki1_278 = arith.constant 6 : i32
        %ki2_278 = arith.constant 8 : i32
        %pad1_278 = arith.constant 1 : i32
        %outidx_278 = arith.constant 278 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_278, %smean_278, %sscale_278, %dilation_278, %padding_278, %ki0_278, %ki1_278, %ki2_278, %pad1_278, %out_v, %outidx_278) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 279: dilation=21 padding=84 kernel=(0,7,8) padding1=0
        %fidx_279 = arith.constant 279 : index
        %bidx_279 = arith.constant 530 : index
        %smidx_279 = arith.constant 950 : index
        %ssidx_279 = arith.constant 1370 : index
        %bias_279 = memref.load %in_v[%bidx_279] : memref<1511xi32>
        %smean_279 = memref.load %in_v[%smidx_279] : memref<1511xi32>
        %sscale_279 = memref.load %in_v[%ssidx_279] : memref<1511xi32>
        %dilation_279 = arith.constant 21 : i32
        %padding_279 = arith.constant 84 : i32
        %ki0_279 = arith.constant 0 : i32
        %ki1_279 = arith.constant 7 : i32
        %ki2_279 = arith.constant 8 : i32
        %pad1_279 = arith.constant 0 : i32
        %outidx_279 = arith.constant 279 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_279, %smean_279, %sscale_279, %dilation_279, %padding_279, %ki0_279, %ki1_279, %ki2_279, %pad1_279, %out_v, %outidx_279) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 280: dilation=21 padding=84 kernel=(1,2,3) padding1=1
        %fidx_280 = arith.constant 280 : index
        %bidx_280 = arith.constant 531 : index
        %smidx_280 = arith.constant 951 : index
        %ssidx_280 = arith.constant 1371 : index
        %bias_280 = memref.load %in_v[%bidx_280] : memref<1511xi32>
        %smean_280 = memref.load %in_v[%smidx_280] : memref<1511xi32>
        %sscale_280 = memref.load %in_v[%ssidx_280] : memref<1511xi32>
        %dilation_280 = arith.constant 21 : i32
        %padding_280 = arith.constant 84 : i32
        %ki0_280 = arith.constant 1 : i32
        %ki1_280 = arith.constant 2 : i32
        %ki2_280 = arith.constant 3 : i32
        %pad1_280 = arith.constant 1 : i32
        %outidx_280 = arith.constant 280 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_280, %smean_280, %sscale_280, %dilation_280, %padding_280, %ki0_280, %ki1_280, %ki2_280, %pad1_280, %out_v, %outidx_280) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 281: dilation=21 padding=84 kernel=(1,2,4) padding1=0
        %fidx_281 = arith.constant 281 : index
        %bidx_281 = arith.constant 532 : index
        %smidx_281 = arith.constant 952 : index
        %ssidx_281 = arith.constant 1372 : index
        %bias_281 = memref.load %in_v[%bidx_281] : memref<1511xi32>
        %smean_281 = memref.load %in_v[%smidx_281] : memref<1511xi32>
        %sscale_281 = memref.load %in_v[%ssidx_281] : memref<1511xi32>
        %dilation_281 = arith.constant 21 : i32
        %padding_281 = arith.constant 84 : i32
        %ki0_281 = arith.constant 1 : i32
        %ki1_281 = arith.constant 2 : i32
        %ki2_281 = arith.constant 4 : i32
        %pad1_281 = arith.constant 0 : i32
        %outidx_281 = arith.constant 281 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_281, %smean_281, %sscale_281, %dilation_281, %padding_281, %ki0_281, %ki1_281, %ki2_281, %pad1_281, %out_v, %outidx_281) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 282: dilation=21 padding=84 kernel=(1,2,5) padding1=1
        %fidx_282 = arith.constant 282 : index
        %bidx_282 = arith.constant 533 : index
        %smidx_282 = arith.constant 953 : index
        %ssidx_282 = arith.constant 1373 : index
        %bias_282 = memref.load %in_v[%bidx_282] : memref<1511xi32>
        %smean_282 = memref.load %in_v[%smidx_282] : memref<1511xi32>
        %sscale_282 = memref.load %in_v[%ssidx_282] : memref<1511xi32>
        %dilation_282 = arith.constant 21 : i32
        %padding_282 = arith.constant 84 : i32
        %ki0_282 = arith.constant 1 : i32
        %ki1_282 = arith.constant 2 : i32
        %ki2_282 = arith.constant 5 : i32
        %pad1_282 = arith.constant 1 : i32
        %outidx_282 = arith.constant 282 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_282, %smean_282, %sscale_282, %dilation_282, %padding_282, %ki0_282, %ki1_282, %ki2_282, %pad1_282, %out_v, %outidx_282) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 283: dilation=21 padding=84 kernel=(1,2,6) padding1=0
        %fidx_283 = arith.constant 283 : index
        %bidx_283 = arith.constant 534 : index
        %smidx_283 = arith.constant 954 : index
        %ssidx_283 = arith.constant 1374 : index
        %bias_283 = memref.load %in_v[%bidx_283] : memref<1511xi32>
        %smean_283 = memref.load %in_v[%smidx_283] : memref<1511xi32>
        %sscale_283 = memref.load %in_v[%ssidx_283] : memref<1511xi32>
        %dilation_283 = arith.constant 21 : i32
        %padding_283 = arith.constant 84 : i32
        %ki0_283 = arith.constant 1 : i32
        %ki1_283 = arith.constant 2 : i32
        %ki2_283 = arith.constant 6 : i32
        %pad1_283 = arith.constant 0 : i32
        %outidx_283 = arith.constant 283 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_283, %smean_283, %sscale_283, %dilation_283, %padding_283, %ki0_283, %ki1_283, %ki2_283, %pad1_283, %out_v, %outidx_283) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 284: dilation=21 padding=84 kernel=(1,2,7) padding1=1
        %fidx_284 = arith.constant 284 : index
        %bidx_284 = arith.constant 535 : index
        %smidx_284 = arith.constant 955 : index
        %ssidx_284 = arith.constant 1375 : index
        %bias_284 = memref.load %in_v[%bidx_284] : memref<1511xi32>
        %smean_284 = memref.load %in_v[%smidx_284] : memref<1511xi32>
        %sscale_284 = memref.load %in_v[%ssidx_284] : memref<1511xi32>
        %dilation_284 = arith.constant 21 : i32
        %padding_284 = arith.constant 84 : i32
        %ki0_284 = arith.constant 1 : i32
        %ki1_284 = arith.constant 2 : i32
        %ki2_284 = arith.constant 7 : i32
        %pad1_284 = arith.constant 1 : i32
        %outidx_284 = arith.constant 284 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_284, %smean_284, %sscale_284, %dilation_284, %padding_284, %ki0_284, %ki1_284, %ki2_284, %pad1_284, %out_v, %outidx_284) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 285: dilation=21 padding=84 kernel=(1,2,8) padding1=0
        %fidx_285 = arith.constant 285 : index
        %bidx_285 = arith.constant 536 : index
        %smidx_285 = arith.constant 956 : index
        %ssidx_285 = arith.constant 1376 : index
        %bias_285 = memref.load %in_v[%bidx_285] : memref<1511xi32>
        %smean_285 = memref.load %in_v[%smidx_285] : memref<1511xi32>
        %sscale_285 = memref.load %in_v[%ssidx_285] : memref<1511xi32>
        %dilation_285 = arith.constant 21 : i32
        %padding_285 = arith.constant 84 : i32
        %ki0_285 = arith.constant 1 : i32
        %ki1_285 = arith.constant 2 : i32
        %ki2_285 = arith.constant 8 : i32
        %pad1_285 = arith.constant 0 : i32
        %outidx_285 = arith.constant 285 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_285, %smean_285, %sscale_285, %dilation_285, %padding_285, %ki0_285, %ki1_285, %ki2_285, %pad1_285, %out_v, %outidx_285) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 286: dilation=21 padding=84 kernel=(1,3,4) padding1=1
        %fidx_286 = arith.constant 286 : index
        %bidx_286 = arith.constant 537 : index
        %smidx_286 = arith.constant 957 : index
        %ssidx_286 = arith.constant 1377 : index
        %bias_286 = memref.load %in_v[%bidx_286] : memref<1511xi32>
        %smean_286 = memref.load %in_v[%smidx_286] : memref<1511xi32>
        %sscale_286 = memref.load %in_v[%ssidx_286] : memref<1511xi32>
        %dilation_286 = arith.constant 21 : i32
        %padding_286 = arith.constant 84 : i32
        %ki0_286 = arith.constant 1 : i32
        %ki1_286 = arith.constant 3 : i32
        %ki2_286 = arith.constant 4 : i32
        %pad1_286 = arith.constant 1 : i32
        %outidx_286 = arith.constant 286 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_286, %smean_286, %sscale_286, %dilation_286, %padding_286, %ki0_286, %ki1_286, %ki2_286, %pad1_286, %out_v, %outidx_286) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 287: dilation=21 padding=84 kernel=(1,3,5) padding1=0
        %fidx_287 = arith.constant 287 : index
        %bidx_287 = arith.constant 538 : index
        %smidx_287 = arith.constant 958 : index
        %ssidx_287 = arith.constant 1378 : index
        %bias_287 = memref.load %in_v[%bidx_287] : memref<1511xi32>
        %smean_287 = memref.load %in_v[%smidx_287] : memref<1511xi32>
        %sscale_287 = memref.load %in_v[%ssidx_287] : memref<1511xi32>
        %dilation_287 = arith.constant 21 : i32
        %padding_287 = arith.constant 84 : i32
        %ki0_287 = arith.constant 1 : i32
        %ki1_287 = arith.constant 3 : i32
        %ki2_287 = arith.constant 5 : i32
        %pad1_287 = arith.constant 0 : i32
        %outidx_287 = arith.constant 287 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_287, %smean_287, %sscale_287, %dilation_287, %padding_287, %ki0_287, %ki1_287, %ki2_287, %pad1_287, %out_v, %outidx_287) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 288: dilation=21 padding=84 kernel=(1,3,6) padding1=1
        %fidx_288 = arith.constant 288 : index
        %bidx_288 = arith.constant 539 : index
        %smidx_288 = arith.constant 959 : index
        %ssidx_288 = arith.constant 1379 : index
        %bias_288 = memref.load %in_v[%bidx_288] : memref<1511xi32>
        %smean_288 = memref.load %in_v[%smidx_288] : memref<1511xi32>
        %sscale_288 = memref.load %in_v[%ssidx_288] : memref<1511xi32>
        %dilation_288 = arith.constant 21 : i32
        %padding_288 = arith.constant 84 : i32
        %ki0_288 = arith.constant 1 : i32
        %ki1_288 = arith.constant 3 : i32
        %ki2_288 = arith.constant 6 : i32
        %pad1_288 = arith.constant 1 : i32
        %outidx_288 = arith.constant 288 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_288, %smean_288, %sscale_288, %dilation_288, %padding_288, %ki0_288, %ki1_288, %ki2_288, %pad1_288, %out_v, %outidx_288) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 289: dilation=21 padding=84 kernel=(1,3,7) padding1=0
        %fidx_289 = arith.constant 289 : index
        %bidx_289 = arith.constant 540 : index
        %smidx_289 = arith.constant 960 : index
        %ssidx_289 = arith.constant 1380 : index
        %bias_289 = memref.load %in_v[%bidx_289] : memref<1511xi32>
        %smean_289 = memref.load %in_v[%smidx_289] : memref<1511xi32>
        %sscale_289 = memref.load %in_v[%ssidx_289] : memref<1511xi32>
        %dilation_289 = arith.constant 21 : i32
        %padding_289 = arith.constant 84 : i32
        %ki0_289 = arith.constant 1 : i32
        %ki1_289 = arith.constant 3 : i32
        %ki2_289 = arith.constant 7 : i32
        %pad1_289 = arith.constant 0 : i32
        %outidx_289 = arith.constant 289 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_289, %smean_289, %sscale_289, %dilation_289, %padding_289, %ki0_289, %ki1_289, %ki2_289, %pad1_289, %out_v, %outidx_289) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 290: dilation=21 padding=84 kernel=(1,3,8) padding1=1
        %fidx_290 = arith.constant 290 : index
        %bidx_290 = arith.constant 541 : index
        %smidx_290 = arith.constant 961 : index
        %ssidx_290 = arith.constant 1381 : index
        %bias_290 = memref.load %in_v[%bidx_290] : memref<1511xi32>
        %smean_290 = memref.load %in_v[%smidx_290] : memref<1511xi32>
        %sscale_290 = memref.load %in_v[%ssidx_290] : memref<1511xi32>
        %dilation_290 = arith.constant 21 : i32
        %padding_290 = arith.constant 84 : i32
        %ki0_290 = arith.constant 1 : i32
        %ki1_290 = arith.constant 3 : i32
        %ki2_290 = arith.constant 8 : i32
        %pad1_290 = arith.constant 1 : i32
        %outidx_290 = arith.constant 290 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_290, %smean_290, %sscale_290, %dilation_290, %padding_290, %ki0_290, %ki1_290, %ki2_290, %pad1_290, %out_v, %outidx_290) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 291: dilation=21 padding=84 kernel=(1,4,5) padding1=0
        %fidx_291 = arith.constant 291 : index
        %bidx_291 = arith.constant 542 : index
        %smidx_291 = arith.constant 962 : index
        %ssidx_291 = arith.constant 1382 : index
        %bias_291 = memref.load %in_v[%bidx_291] : memref<1511xi32>
        %smean_291 = memref.load %in_v[%smidx_291] : memref<1511xi32>
        %sscale_291 = memref.load %in_v[%ssidx_291] : memref<1511xi32>
        %dilation_291 = arith.constant 21 : i32
        %padding_291 = arith.constant 84 : i32
        %ki0_291 = arith.constant 1 : i32
        %ki1_291 = arith.constant 4 : i32
        %ki2_291 = arith.constant 5 : i32
        %pad1_291 = arith.constant 0 : i32
        %outidx_291 = arith.constant 291 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_291, %smean_291, %sscale_291, %dilation_291, %padding_291, %ki0_291, %ki1_291, %ki2_291, %pad1_291, %out_v, %outidx_291) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 292: dilation=21 padding=84 kernel=(1,4,6) padding1=1
        %fidx_292 = arith.constant 292 : index
        %bidx_292 = arith.constant 543 : index
        %smidx_292 = arith.constant 963 : index
        %ssidx_292 = arith.constant 1383 : index
        %bias_292 = memref.load %in_v[%bidx_292] : memref<1511xi32>
        %smean_292 = memref.load %in_v[%smidx_292] : memref<1511xi32>
        %sscale_292 = memref.load %in_v[%ssidx_292] : memref<1511xi32>
        %dilation_292 = arith.constant 21 : i32
        %padding_292 = arith.constant 84 : i32
        %ki0_292 = arith.constant 1 : i32
        %ki1_292 = arith.constant 4 : i32
        %ki2_292 = arith.constant 6 : i32
        %pad1_292 = arith.constant 1 : i32
        %outidx_292 = arith.constant 292 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_292, %smean_292, %sscale_292, %dilation_292, %padding_292, %ki0_292, %ki1_292, %ki2_292, %pad1_292, %out_v, %outidx_292) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 293: dilation=21 padding=84 kernel=(1,4,7) padding1=0
        %fidx_293 = arith.constant 293 : index
        %bidx_293 = arith.constant 544 : index
        %smidx_293 = arith.constant 964 : index
        %ssidx_293 = arith.constant 1384 : index
        %bias_293 = memref.load %in_v[%bidx_293] : memref<1511xi32>
        %smean_293 = memref.load %in_v[%smidx_293] : memref<1511xi32>
        %sscale_293 = memref.load %in_v[%ssidx_293] : memref<1511xi32>
        %dilation_293 = arith.constant 21 : i32
        %padding_293 = arith.constant 84 : i32
        %ki0_293 = arith.constant 1 : i32
        %ki1_293 = arith.constant 4 : i32
        %ki2_293 = arith.constant 7 : i32
        %pad1_293 = arith.constant 0 : i32
        %outidx_293 = arith.constant 293 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_293, %smean_293, %sscale_293, %dilation_293, %padding_293, %ki0_293, %ki1_293, %ki2_293, %pad1_293, %out_v, %outidx_293) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 294: dilation=21 padding=84 kernel=(1,4,8) padding1=1
        %fidx_294 = arith.constant 294 : index
        %bidx_294 = arith.constant 545 : index
        %smidx_294 = arith.constant 965 : index
        %ssidx_294 = arith.constant 1385 : index
        %bias_294 = memref.load %in_v[%bidx_294] : memref<1511xi32>
        %smean_294 = memref.load %in_v[%smidx_294] : memref<1511xi32>
        %sscale_294 = memref.load %in_v[%ssidx_294] : memref<1511xi32>
        %dilation_294 = arith.constant 21 : i32
        %padding_294 = arith.constant 84 : i32
        %ki0_294 = arith.constant 1 : i32
        %ki1_294 = arith.constant 4 : i32
        %ki2_294 = arith.constant 8 : i32
        %pad1_294 = arith.constant 1 : i32
        %outidx_294 = arith.constant 294 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_294, %smean_294, %sscale_294, %dilation_294, %padding_294, %ki0_294, %ki1_294, %ki2_294, %pad1_294, %out_v, %outidx_294) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 295: dilation=21 padding=84 kernel=(1,5,6) padding1=0
        %fidx_295 = arith.constant 295 : index
        %bidx_295 = arith.constant 546 : index
        %smidx_295 = arith.constant 966 : index
        %ssidx_295 = arith.constant 1386 : index
        %bias_295 = memref.load %in_v[%bidx_295] : memref<1511xi32>
        %smean_295 = memref.load %in_v[%smidx_295] : memref<1511xi32>
        %sscale_295 = memref.load %in_v[%ssidx_295] : memref<1511xi32>
        %dilation_295 = arith.constant 21 : i32
        %padding_295 = arith.constant 84 : i32
        %ki0_295 = arith.constant 1 : i32
        %ki1_295 = arith.constant 5 : i32
        %ki2_295 = arith.constant 6 : i32
        %pad1_295 = arith.constant 0 : i32
        %outidx_295 = arith.constant 295 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_295, %smean_295, %sscale_295, %dilation_295, %padding_295, %ki0_295, %ki1_295, %ki2_295, %pad1_295, %out_v, %outidx_295) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 296: dilation=21 padding=84 kernel=(1,5,7) padding1=1
        %fidx_296 = arith.constant 296 : index
        %bidx_296 = arith.constant 547 : index
        %smidx_296 = arith.constant 967 : index
        %ssidx_296 = arith.constant 1387 : index
        %bias_296 = memref.load %in_v[%bidx_296] : memref<1511xi32>
        %smean_296 = memref.load %in_v[%smidx_296] : memref<1511xi32>
        %sscale_296 = memref.load %in_v[%ssidx_296] : memref<1511xi32>
        %dilation_296 = arith.constant 21 : i32
        %padding_296 = arith.constant 84 : i32
        %ki0_296 = arith.constant 1 : i32
        %ki1_296 = arith.constant 5 : i32
        %ki2_296 = arith.constant 7 : i32
        %pad1_296 = arith.constant 1 : i32
        %outidx_296 = arith.constant 296 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_296, %smean_296, %sscale_296, %dilation_296, %padding_296, %ki0_296, %ki1_296, %ki2_296, %pad1_296, %out_v, %outidx_296) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 297: dilation=21 padding=84 kernel=(1,5,8) padding1=0
        %fidx_297 = arith.constant 297 : index
        %bidx_297 = arith.constant 548 : index
        %smidx_297 = arith.constant 968 : index
        %ssidx_297 = arith.constant 1388 : index
        %bias_297 = memref.load %in_v[%bidx_297] : memref<1511xi32>
        %smean_297 = memref.load %in_v[%smidx_297] : memref<1511xi32>
        %sscale_297 = memref.load %in_v[%ssidx_297] : memref<1511xi32>
        %dilation_297 = arith.constant 21 : i32
        %padding_297 = arith.constant 84 : i32
        %ki0_297 = arith.constant 1 : i32
        %ki1_297 = arith.constant 5 : i32
        %ki2_297 = arith.constant 8 : i32
        %pad1_297 = arith.constant 0 : i32
        %outidx_297 = arith.constant 297 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_297, %smean_297, %sscale_297, %dilation_297, %padding_297, %ki0_297, %ki1_297, %ki2_297, %pad1_297, %out_v, %outidx_297) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 298: dilation=21 padding=84 kernel=(1,6,7) padding1=1
        %fidx_298 = arith.constant 298 : index
        %bidx_298 = arith.constant 549 : index
        %smidx_298 = arith.constant 969 : index
        %ssidx_298 = arith.constant 1389 : index
        %bias_298 = memref.load %in_v[%bidx_298] : memref<1511xi32>
        %smean_298 = memref.load %in_v[%smidx_298] : memref<1511xi32>
        %sscale_298 = memref.load %in_v[%ssidx_298] : memref<1511xi32>
        %dilation_298 = arith.constant 21 : i32
        %padding_298 = arith.constant 84 : i32
        %ki0_298 = arith.constant 1 : i32
        %ki1_298 = arith.constant 6 : i32
        %ki2_298 = arith.constant 7 : i32
        %pad1_298 = arith.constant 1 : i32
        %outidx_298 = arith.constant 298 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_298, %smean_298, %sscale_298, %dilation_298, %padding_298, %ki0_298, %ki1_298, %ki2_298, %pad1_298, %out_v, %outidx_298) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 299: dilation=21 padding=84 kernel=(1,6,8) padding1=0
        %fidx_299 = arith.constant 299 : index
        %bidx_299 = arith.constant 550 : index
        %smidx_299 = arith.constant 970 : index
        %ssidx_299 = arith.constant 1390 : index
        %bias_299 = memref.load %in_v[%bidx_299] : memref<1511xi32>
        %smean_299 = memref.load %in_v[%smidx_299] : memref<1511xi32>
        %sscale_299 = memref.load %in_v[%ssidx_299] : memref<1511xi32>
        %dilation_299 = arith.constant 21 : i32
        %padding_299 = arith.constant 84 : i32
        %ki0_299 = arith.constant 1 : i32
        %ki1_299 = arith.constant 6 : i32
        %ki2_299 = arith.constant 8 : i32
        %pad1_299 = arith.constant 0 : i32
        %outidx_299 = arith.constant 299 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_299, %smean_299, %sscale_299, %dilation_299, %padding_299, %ki0_299, %ki1_299, %ki2_299, %pad1_299, %out_v, %outidx_299) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 300: dilation=21 padding=84 kernel=(1,7,8) padding1=1
        %fidx_300 = arith.constant 300 : index
        %bidx_300 = arith.constant 551 : index
        %smidx_300 = arith.constant 971 : index
        %ssidx_300 = arith.constant 1391 : index
        %bias_300 = memref.load %in_v[%bidx_300] : memref<1511xi32>
        %smean_300 = memref.load %in_v[%smidx_300] : memref<1511xi32>
        %sscale_300 = memref.load %in_v[%ssidx_300] : memref<1511xi32>
        %dilation_300 = arith.constant 21 : i32
        %padding_300 = arith.constant 84 : i32
        %ki0_300 = arith.constant 1 : i32
        %ki1_300 = arith.constant 7 : i32
        %ki2_300 = arith.constant 8 : i32
        %pad1_300 = arith.constant 1 : i32
        %outidx_300 = arith.constant 300 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_300, %smean_300, %sscale_300, %dilation_300, %padding_300, %ki0_300, %ki1_300, %ki2_300, %pad1_300, %out_v, %outidx_300) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 301: dilation=21 padding=84 kernel=(2,3,4) padding1=0
        %fidx_301 = arith.constant 301 : index
        %bidx_301 = arith.constant 552 : index
        %smidx_301 = arith.constant 972 : index
        %ssidx_301 = arith.constant 1392 : index
        %bias_301 = memref.load %in_v[%bidx_301] : memref<1511xi32>
        %smean_301 = memref.load %in_v[%smidx_301] : memref<1511xi32>
        %sscale_301 = memref.load %in_v[%ssidx_301] : memref<1511xi32>
        %dilation_301 = arith.constant 21 : i32
        %padding_301 = arith.constant 84 : i32
        %ki0_301 = arith.constant 2 : i32
        %ki1_301 = arith.constant 3 : i32
        %ki2_301 = arith.constant 4 : i32
        %pad1_301 = arith.constant 0 : i32
        %outidx_301 = arith.constant 301 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_301, %smean_301, %sscale_301, %dilation_301, %padding_301, %ki0_301, %ki1_301, %ki2_301, %pad1_301, %out_v, %outidx_301) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 302: dilation=21 padding=84 kernel=(2,3,5) padding1=1
        %fidx_302 = arith.constant 302 : index
        %bidx_302 = arith.constant 553 : index
        %smidx_302 = arith.constant 973 : index
        %ssidx_302 = arith.constant 1393 : index
        %bias_302 = memref.load %in_v[%bidx_302] : memref<1511xi32>
        %smean_302 = memref.load %in_v[%smidx_302] : memref<1511xi32>
        %sscale_302 = memref.load %in_v[%ssidx_302] : memref<1511xi32>
        %dilation_302 = arith.constant 21 : i32
        %padding_302 = arith.constant 84 : i32
        %ki0_302 = arith.constant 2 : i32
        %ki1_302 = arith.constant 3 : i32
        %ki2_302 = arith.constant 5 : i32
        %pad1_302 = arith.constant 1 : i32
        %outidx_302 = arith.constant 302 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_302, %smean_302, %sscale_302, %dilation_302, %padding_302, %ki0_302, %ki1_302, %ki2_302, %pad1_302, %out_v, %outidx_302) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 303: dilation=21 padding=84 kernel=(2,3,6) padding1=0
        %fidx_303 = arith.constant 303 : index
        %bidx_303 = arith.constant 554 : index
        %smidx_303 = arith.constant 974 : index
        %ssidx_303 = arith.constant 1394 : index
        %bias_303 = memref.load %in_v[%bidx_303] : memref<1511xi32>
        %smean_303 = memref.load %in_v[%smidx_303] : memref<1511xi32>
        %sscale_303 = memref.load %in_v[%ssidx_303] : memref<1511xi32>
        %dilation_303 = arith.constant 21 : i32
        %padding_303 = arith.constant 84 : i32
        %ki0_303 = arith.constant 2 : i32
        %ki1_303 = arith.constant 3 : i32
        %ki2_303 = arith.constant 6 : i32
        %pad1_303 = arith.constant 0 : i32
        %outidx_303 = arith.constant 303 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_303, %smean_303, %sscale_303, %dilation_303, %padding_303, %ki0_303, %ki1_303, %ki2_303, %pad1_303, %out_v, %outidx_303) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 304: dilation=21 padding=84 kernel=(2,3,7) padding1=1
        %fidx_304 = arith.constant 304 : index
        %bidx_304 = arith.constant 555 : index
        %smidx_304 = arith.constant 975 : index
        %ssidx_304 = arith.constant 1395 : index
        %bias_304 = memref.load %in_v[%bidx_304] : memref<1511xi32>
        %smean_304 = memref.load %in_v[%smidx_304] : memref<1511xi32>
        %sscale_304 = memref.load %in_v[%ssidx_304] : memref<1511xi32>
        %dilation_304 = arith.constant 21 : i32
        %padding_304 = arith.constant 84 : i32
        %ki0_304 = arith.constant 2 : i32
        %ki1_304 = arith.constant 3 : i32
        %ki2_304 = arith.constant 7 : i32
        %pad1_304 = arith.constant 1 : i32
        %outidx_304 = arith.constant 304 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_304, %smean_304, %sscale_304, %dilation_304, %padding_304, %ki0_304, %ki1_304, %ki2_304, %pad1_304, %out_v, %outidx_304) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 305: dilation=21 padding=84 kernel=(2,3,8) padding1=0
        %fidx_305 = arith.constant 305 : index
        %bidx_305 = arith.constant 556 : index
        %smidx_305 = arith.constant 976 : index
        %ssidx_305 = arith.constant 1396 : index
        %bias_305 = memref.load %in_v[%bidx_305] : memref<1511xi32>
        %smean_305 = memref.load %in_v[%smidx_305] : memref<1511xi32>
        %sscale_305 = memref.load %in_v[%ssidx_305] : memref<1511xi32>
        %dilation_305 = arith.constant 21 : i32
        %padding_305 = arith.constant 84 : i32
        %ki0_305 = arith.constant 2 : i32
        %ki1_305 = arith.constant 3 : i32
        %ki2_305 = arith.constant 8 : i32
        %pad1_305 = arith.constant 0 : i32
        %outidx_305 = arith.constant 305 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_305, %smean_305, %sscale_305, %dilation_305, %padding_305, %ki0_305, %ki1_305, %ki2_305, %pad1_305, %out_v, %outidx_305) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 306: dilation=21 padding=84 kernel=(2,4,5) padding1=1
        %fidx_306 = arith.constant 306 : index
        %bidx_306 = arith.constant 557 : index
        %smidx_306 = arith.constant 977 : index
        %ssidx_306 = arith.constant 1397 : index
        %bias_306 = memref.load %in_v[%bidx_306] : memref<1511xi32>
        %smean_306 = memref.load %in_v[%smidx_306] : memref<1511xi32>
        %sscale_306 = memref.load %in_v[%ssidx_306] : memref<1511xi32>
        %dilation_306 = arith.constant 21 : i32
        %padding_306 = arith.constant 84 : i32
        %ki0_306 = arith.constant 2 : i32
        %ki1_306 = arith.constant 4 : i32
        %ki2_306 = arith.constant 5 : i32
        %pad1_306 = arith.constant 1 : i32
        %outidx_306 = arith.constant 306 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_306, %smean_306, %sscale_306, %dilation_306, %padding_306, %ki0_306, %ki1_306, %ki2_306, %pad1_306, %out_v, %outidx_306) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 307: dilation=21 padding=84 kernel=(2,4,6) padding1=0
        %fidx_307 = arith.constant 307 : index
        %bidx_307 = arith.constant 558 : index
        %smidx_307 = arith.constant 978 : index
        %ssidx_307 = arith.constant 1398 : index
        %bias_307 = memref.load %in_v[%bidx_307] : memref<1511xi32>
        %smean_307 = memref.load %in_v[%smidx_307] : memref<1511xi32>
        %sscale_307 = memref.load %in_v[%ssidx_307] : memref<1511xi32>
        %dilation_307 = arith.constant 21 : i32
        %padding_307 = arith.constant 84 : i32
        %ki0_307 = arith.constant 2 : i32
        %ki1_307 = arith.constant 4 : i32
        %ki2_307 = arith.constant 6 : i32
        %pad1_307 = arith.constant 0 : i32
        %outidx_307 = arith.constant 307 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_307, %smean_307, %sscale_307, %dilation_307, %padding_307, %ki0_307, %ki1_307, %ki2_307, %pad1_307, %out_v, %outidx_307) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 308: dilation=21 padding=84 kernel=(2,4,7) padding1=1
        %fidx_308 = arith.constant 308 : index
        %bidx_308 = arith.constant 559 : index
        %smidx_308 = arith.constant 979 : index
        %ssidx_308 = arith.constant 1399 : index
        %bias_308 = memref.load %in_v[%bidx_308] : memref<1511xi32>
        %smean_308 = memref.load %in_v[%smidx_308] : memref<1511xi32>
        %sscale_308 = memref.load %in_v[%ssidx_308] : memref<1511xi32>
        %dilation_308 = arith.constant 21 : i32
        %padding_308 = arith.constant 84 : i32
        %ki0_308 = arith.constant 2 : i32
        %ki1_308 = arith.constant 4 : i32
        %ki2_308 = arith.constant 7 : i32
        %pad1_308 = arith.constant 1 : i32
        %outidx_308 = arith.constant 308 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_308, %smean_308, %sscale_308, %dilation_308, %padding_308, %ki0_308, %ki1_308, %ki2_308, %pad1_308, %out_v, %outidx_308) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 309: dilation=21 padding=84 kernel=(2,4,8) padding1=0
        %fidx_309 = arith.constant 309 : index
        %bidx_309 = arith.constant 560 : index
        %smidx_309 = arith.constant 980 : index
        %ssidx_309 = arith.constant 1400 : index
        %bias_309 = memref.load %in_v[%bidx_309] : memref<1511xi32>
        %smean_309 = memref.load %in_v[%smidx_309] : memref<1511xi32>
        %sscale_309 = memref.load %in_v[%ssidx_309] : memref<1511xi32>
        %dilation_309 = arith.constant 21 : i32
        %padding_309 = arith.constant 84 : i32
        %ki0_309 = arith.constant 2 : i32
        %ki1_309 = arith.constant 4 : i32
        %ki2_309 = arith.constant 8 : i32
        %pad1_309 = arith.constant 0 : i32
        %outidx_309 = arith.constant 309 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_309, %smean_309, %sscale_309, %dilation_309, %padding_309, %ki0_309, %ki1_309, %ki2_309, %pad1_309, %out_v, %outidx_309) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 310: dilation=21 padding=84 kernel=(2,5,6) padding1=1
        %fidx_310 = arith.constant 310 : index
        %bidx_310 = arith.constant 561 : index
        %smidx_310 = arith.constant 981 : index
        %ssidx_310 = arith.constant 1401 : index
        %bias_310 = memref.load %in_v[%bidx_310] : memref<1511xi32>
        %smean_310 = memref.load %in_v[%smidx_310] : memref<1511xi32>
        %sscale_310 = memref.load %in_v[%ssidx_310] : memref<1511xi32>
        %dilation_310 = arith.constant 21 : i32
        %padding_310 = arith.constant 84 : i32
        %ki0_310 = arith.constant 2 : i32
        %ki1_310 = arith.constant 5 : i32
        %ki2_310 = arith.constant 6 : i32
        %pad1_310 = arith.constant 1 : i32
        %outidx_310 = arith.constant 310 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_310, %smean_310, %sscale_310, %dilation_310, %padding_310, %ki0_310, %ki1_310, %ki2_310, %pad1_310, %out_v, %outidx_310) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 311: dilation=21 padding=84 kernel=(2,5,7) padding1=0
        %fidx_311 = arith.constant 311 : index
        %bidx_311 = arith.constant 562 : index
        %smidx_311 = arith.constant 982 : index
        %ssidx_311 = arith.constant 1402 : index
        %bias_311 = memref.load %in_v[%bidx_311] : memref<1511xi32>
        %smean_311 = memref.load %in_v[%smidx_311] : memref<1511xi32>
        %sscale_311 = memref.load %in_v[%ssidx_311] : memref<1511xi32>
        %dilation_311 = arith.constant 21 : i32
        %padding_311 = arith.constant 84 : i32
        %ki0_311 = arith.constant 2 : i32
        %ki1_311 = arith.constant 5 : i32
        %ki2_311 = arith.constant 7 : i32
        %pad1_311 = arith.constant 0 : i32
        %outidx_311 = arith.constant 311 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_311, %smean_311, %sscale_311, %dilation_311, %padding_311, %ki0_311, %ki1_311, %ki2_311, %pad1_311, %out_v, %outidx_311) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 312: dilation=21 padding=84 kernel=(2,5,8) padding1=1
        %fidx_312 = arith.constant 312 : index
        %bidx_312 = arith.constant 563 : index
        %smidx_312 = arith.constant 983 : index
        %ssidx_312 = arith.constant 1403 : index
        %bias_312 = memref.load %in_v[%bidx_312] : memref<1511xi32>
        %smean_312 = memref.load %in_v[%smidx_312] : memref<1511xi32>
        %sscale_312 = memref.load %in_v[%ssidx_312] : memref<1511xi32>
        %dilation_312 = arith.constant 21 : i32
        %padding_312 = arith.constant 84 : i32
        %ki0_312 = arith.constant 2 : i32
        %ki1_312 = arith.constant 5 : i32
        %ki2_312 = arith.constant 8 : i32
        %pad1_312 = arith.constant 1 : i32
        %outidx_312 = arith.constant 312 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_312, %smean_312, %sscale_312, %dilation_312, %padding_312, %ki0_312, %ki1_312, %ki2_312, %pad1_312, %out_v, %outidx_312) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 313: dilation=21 padding=84 kernel=(2,6,7) padding1=0
        %fidx_313 = arith.constant 313 : index
        %bidx_313 = arith.constant 564 : index
        %smidx_313 = arith.constant 984 : index
        %ssidx_313 = arith.constant 1404 : index
        %bias_313 = memref.load %in_v[%bidx_313] : memref<1511xi32>
        %smean_313 = memref.load %in_v[%smidx_313] : memref<1511xi32>
        %sscale_313 = memref.load %in_v[%ssidx_313] : memref<1511xi32>
        %dilation_313 = arith.constant 21 : i32
        %padding_313 = arith.constant 84 : i32
        %ki0_313 = arith.constant 2 : i32
        %ki1_313 = arith.constant 6 : i32
        %ki2_313 = arith.constant 7 : i32
        %pad1_313 = arith.constant 0 : i32
        %outidx_313 = arith.constant 313 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_313, %smean_313, %sscale_313, %dilation_313, %padding_313, %ki0_313, %ki1_313, %ki2_313, %pad1_313, %out_v, %outidx_313) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 314: dilation=21 padding=84 kernel=(2,6,8) padding1=1
        %fidx_314 = arith.constant 314 : index
        %bidx_314 = arith.constant 565 : index
        %smidx_314 = arith.constant 985 : index
        %ssidx_314 = arith.constant 1405 : index
        %bias_314 = memref.load %in_v[%bidx_314] : memref<1511xi32>
        %smean_314 = memref.load %in_v[%smidx_314] : memref<1511xi32>
        %sscale_314 = memref.load %in_v[%ssidx_314] : memref<1511xi32>
        %dilation_314 = arith.constant 21 : i32
        %padding_314 = arith.constant 84 : i32
        %ki0_314 = arith.constant 2 : i32
        %ki1_314 = arith.constant 6 : i32
        %ki2_314 = arith.constant 8 : i32
        %pad1_314 = arith.constant 1 : i32
        %outidx_314 = arith.constant 314 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_314, %smean_314, %sscale_314, %dilation_314, %padding_314, %ki0_314, %ki1_314, %ki2_314, %pad1_314, %out_v, %outidx_314) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 315: dilation=21 padding=84 kernel=(2,7,8) padding1=0
        %fidx_315 = arith.constant 315 : index
        %bidx_315 = arith.constant 566 : index
        %smidx_315 = arith.constant 986 : index
        %ssidx_315 = arith.constant 1406 : index
        %bias_315 = memref.load %in_v[%bidx_315] : memref<1511xi32>
        %smean_315 = memref.load %in_v[%smidx_315] : memref<1511xi32>
        %sscale_315 = memref.load %in_v[%ssidx_315] : memref<1511xi32>
        %dilation_315 = arith.constant 21 : i32
        %padding_315 = arith.constant 84 : i32
        %ki0_315 = arith.constant 2 : i32
        %ki1_315 = arith.constant 7 : i32
        %ki2_315 = arith.constant 8 : i32
        %pad1_315 = arith.constant 0 : i32
        %outidx_315 = arith.constant 315 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_315, %smean_315, %sscale_315, %dilation_315, %padding_315, %ki0_315, %ki1_315, %ki2_315, %pad1_315, %out_v, %outidx_315) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 316: dilation=21 padding=84 kernel=(3,4,5) padding1=1
        %fidx_316 = arith.constant 316 : index
        %bidx_316 = arith.constant 567 : index
        %smidx_316 = arith.constant 987 : index
        %ssidx_316 = arith.constant 1407 : index
        %bias_316 = memref.load %in_v[%bidx_316] : memref<1511xi32>
        %smean_316 = memref.load %in_v[%smidx_316] : memref<1511xi32>
        %sscale_316 = memref.load %in_v[%ssidx_316] : memref<1511xi32>
        %dilation_316 = arith.constant 21 : i32
        %padding_316 = arith.constant 84 : i32
        %ki0_316 = arith.constant 3 : i32
        %ki1_316 = arith.constant 4 : i32
        %ki2_316 = arith.constant 5 : i32
        %pad1_316 = arith.constant 1 : i32
        %outidx_316 = arith.constant 316 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_316, %smean_316, %sscale_316, %dilation_316, %padding_316, %ki0_316, %ki1_316, %ki2_316, %pad1_316, %out_v, %outidx_316) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 317: dilation=21 padding=84 kernel=(3,4,6) padding1=0
        %fidx_317 = arith.constant 317 : index
        %bidx_317 = arith.constant 568 : index
        %smidx_317 = arith.constant 988 : index
        %ssidx_317 = arith.constant 1408 : index
        %bias_317 = memref.load %in_v[%bidx_317] : memref<1511xi32>
        %smean_317 = memref.load %in_v[%smidx_317] : memref<1511xi32>
        %sscale_317 = memref.load %in_v[%ssidx_317] : memref<1511xi32>
        %dilation_317 = arith.constant 21 : i32
        %padding_317 = arith.constant 84 : i32
        %ki0_317 = arith.constant 3 : i32
        %ki1_317 = arith.constant 4 : i32
        %ki2_317 = arith.constant 6 : i32
        %pad1_317 = arith.constant 0 : i32
        %outidx_317 = arith.constant 317 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_317, %smean_317, %sscale_317, %dilation_317, %padding_317, %ki0_317, %ki1_317, %ki2_317, %pad1_317, %out_v, %outidx_317) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 318: dilation=21 padding=84 kernel=(3,4,7) padding1=1
        %fidx_318 = arith.constant 318 : index
        %bidx_318 = arith.constant 569 : index
        %smidx_318 = arith.constant 989 : index
        %ssidx_318 = arith.constant 1409 : index
        %bias_318 = memref.load %in_v[%bidx_318] : memref<1511xi32>
        %smean_318 = memref.load %in_v[%smidx_318] : memref<1511xi32>
        %sscale_318 = memref.load %in_v[%ssidx_318] : memref<1511xi32>
        %dilation_318 = arith.constant 21 : i32
        %padding_318 = arith.constant 84 : i32
        %ki0_318 = arith.constant 3 : i32
        %ki1_318 = arith.constant 4 : i32
        %ki2_318 = arith.constant 7 : i32
        %pad1_318 = arith.constant 1 : i32
        %outidx_318 = arith.constant 318 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_318, %smean_318, %sscale_318, %dilation_318, %padding_318, %ki0_318, %ki1_318, %ki2_318, %pad1_318, %out_v, %outidx_318) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 319: dilation=21 padding=84 kernel=(3,4,8) padding1=0
        %fidx_319 = arith.constant 319 : index
        %bidx_319 = arith.constant 570 : index
        %smidx_319 = arith.constant 990 : index
        %ssidx_319 = arith.constant 1410 : index
        %bias_319 = memref.load %in_v[%bidx_319] : memref<1511xi32>
        %smean_319 = memref.load %in_v[%smidx_319] : memref<1511xi32>
        %sscale_319 = memref.load %in_v[%ssidx_319] : memref<1511xi32>
        %dilation_319 = arith.constant 21 : i32
        %padding_319 = arith.constant 84 : i32
        %ki0_319 = arith.constant 3 : i32
        %ki1_319 = arith.constant 4 : i32
        %ki2_319 = arith.constant 8 : i32
        %pad1_319 = arith.constant 0 : i32
        %outidx_319 = arith.constant 319 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_319, %smean_319, %sscale_319, %dilation_319, %padding_319, %ki0_319, %ki1_319, %ki2_319, %pad1_319, %out_v, %outidx_319) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 320: dilation=21 padding=84 kernel=(3,5,6) padding1=1
        %fidx_320 = arith.constant 320 : index
        %bidx_320 = arith.constant 571 : index
        %smidx_320 = arith.constant 991 : index
        %ssidx_320 = arith.constant 1411 : index
        %bias_320 = memref.load %in_v[%bidx_320] : memref<1511xi32>
        %smean_320 = memref.load %in_v[%smidx_320] : memref<1511xi32>
        %sscale_320 = memref.load %in_v[%ssidx_320] : memref<1511xi32>
        %dilation_320 = arith.constant 21 : i32
        %padding_320 = arith.constant 84 : i32
        %ki0_320 = arith.constant 3 : i32
        %ki1_320 = arith.constant 5 : i32
        %ki2_320 = arith.constant 6 : i32
        %pad1_320 = arith.constant 1 : i32
        %outidx_320 = arith.constant 320 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_320, %smean_320, %sscale_320, %dilation_320, %padding_320, %ki0_320, %ki1_320, %ki2_320, %pad1_320, %out_v, %outidx_320) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 321: dilation=21 padding=84 kernel=(3,5,7) padding1=0
        %fidx_321 = arith.constant 321 : index
        %bidx_321 = arith.constant 572 : index
        %smidx_321 = arith.constant 992 : index
        %ssidx_321 = arith.constant 1412 : index
        %bias_321 = memref.load %in_v[%bidx_321] : memref<1511xi32>
        %smean_321 = memref.load %in_v[%smidx_321] : memref<1511xi32>
        %sscale_321 = memref.load %in_v[%ssidx_321] : memref<1511xi32>
        %dilation_321 = arith.constant 21 : i32
        %padding_321 = arith.constant 84 : i32
        %ki0_321 = arith.constant 3 : i32
        %ki1_321 = arith.constant 5 : i32
        %ki2_321 = arith.constant 7 : i32
        %pad1_321 = arith.constant 0 : i32
        %outidx_321 = arith.constant 321 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_321, %smean_321, %sscale_321, %dilation_321, %padding_321, %ki0_321, %ki1_321, %ki2_321, %pad1_321, %out_v, %outidx_321) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 322: dilation=21 padding=84 kernel=(3,5,8) padding1=1
        %fidx_322 = arith.constant 322 : index
        %bidx_322 = arith.constant 573 : index
        %smidx_322 = arith.constant 993 : index
        %ssidx_322 = arith.constant 1413 : index
        %bias_322 = memref.load %in_v[%bidx_322] : memref<1511xi32>
        %smean_322 = memref.load %in_v[%smidx_322] : memref<1511xi32>
        %sscale_322 = memref.load %in_v[%ssidx_322] : memref<1511xi32>
        %dilation_322 = arith.constant 21 : i32
        %padding_322 = arith.constant 84 : i32
        %ki0_322 = arith.constant 3 : i32
        %ki1_322 = arith.constant 5 : i32
        %ki2_322 = arith.constant 8 : i32
        %pad1_322 = arith.constant 1 : i32
        %outidx_322 = arith.constant 322 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_322, %smean_322, %sscale_322, %dilation_322, %padding_322, %ki0_322, %ki1_322, %ki2_322, %pad1_322, %out_v, %outidx_322) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 323: dilation=21 padding=84 kernel=(3,6,7) padding1=0
        %fidx_323 = arith.constant 323 : index
        %bidx_323 = arith.constant 574 : index
        %smidx_323 = arith.constant 994 : index
        %ssidx_323 = arith.constant 1414 : index
        %bias_323 = memref.load %in_v[%bidx_323] : memref<1511xi32>
        %smean_323 = memref.load %in_v[%smidx_323] : memref<1511xi32>
        %sscale_323 = memref.load %in_v[%ssidx_323] : memref<1511xi32>
        %dilation_323 = arith.constant 21 : i32
        %padding_323 = arith.constant 84 : i32
        %ki0_323 = arith.constant 3 : i32
        %ki1_323 = arith.constant 6 : i32
        %ki2_323 = arith.constant 7 : i32
        %pad1_323 = arith.constant 0 : i32
        %outidx_323 = arith.constant 323 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_323, %smean_323, %sscale_323, %dilation_323, %padding_323, %ki0_323, %ki1_323, %ki2_323, %pad1_323, %out_v, %outidx_323) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 324: dilation=21 padding=84 kernel=(3,6,8) padding1=1
        %fidx_324 = arith.constant 324 : index
        %bidx_324 = arith.constant 575 : index
        %smidx_324 = arith.constant 995 : index
        %ssidx_324 = arith.constant 1415 : index
        %bias_324 = memref.load %in_v[%bidx_324] : memref<1511xi32>
        %smean_324 = memref.load %in_v[%smidx_324] : memref<1511xi32>
        %sscale_324 = memref.load %in_v[%ssidx_324] : memref<1511xi32>
        %dilation_324 = arith.constant 21 : i32
        %padding_324 = arith.constant 84 : i32
        %ki0_324 = arith.constant 3 : i32
        %ki1_324 = arith.constant 6 : i32
        %ki2_324 = arith.constant 8 : i32
        %pad1_324 = arith.constant 1 : i32
        %outidx_324 = arith.constant 324 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_324, %smean_324, %sscale_324, %dilation_324, %padding_324, %ki0_324, %ki1_324, %ki2_324, %pad1_324, %out_v, %outidx_324) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 325: dilation=21 padding=84 kernel=(3,7,8) padding1=0
        %fidx_325 = arith.constant 325 : index
        %bidx_325 = arith.constant 576 : index
        %smidx_325 = arith.constant 996 : index
        %ssidx_325 = arith.constant 1416 : index
        %bias_325 = memref.load %in_v[%bidx_325] : memref<1511xi32>
        %smean_325 = memref.load %in_v[%smidx_325] : memref<1511xi32>
        %sscale_325 = memref.load %in_v[%ssidx_325] : memref<1511xi32>
        %dilation_325 = arith.constant 21 : i32
        %padding_325 = arith.constant 84 : i32
        %ki0_325 = arith.constant 3 : i32
        %ki1_325 = arith.constant 7 : i32
        %ki2_325 = arith.constant 8 : i32
        %pad1_325 = arith.constant 0 : i32
        %outidx_325 = arith.constant 325 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_325, %smean_325, %sscale_325, %dilation_325, %padding_325, %ki0_325, %ki1_325, %ki2_325, %pad1_325, %out_v, %outidx_325) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 326: dilation=21 padding=84 kernel=(4,5,6) padding1=1
        %fidx_326 = arith.constant 326 : index
        %bidx_326 = arith.constant 577 : index
        %smidx_326 = arith.constant 997 : index
        %ssidx_326 = arith.constant 1417 : index
        %bias_326 = memref.load %in_v[%bidx_326] : memref<1511xi32>
        %smean_326 = memref.load %in_v[%smidx_326] : memref<1511xi32>
        %sscale_326 = memref.load %in_v[%ssidx_326] : memref<1511xi32>
        %dilation_326 = arith.constant 21 : i32
        %padding_326 = arith.constant 84 : i32
        %ki0_326 = arith.constant 4 : i32
        %ki1_326 = arith.constant 5 : i32
        %ki2_326 = arith.constant 6 : i32
        %pad1_326 = arith.constant 1 : i32
        %outidx_326 = arith.constant 326 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_326, %smean_326, %sscale_326, %dilation_326, %padding_326, %ki0_326, %ki1_326, %ki2_326, %pad1_326, %out_v, %outidx_326) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 327: dilation=21 padding=84 kernel=(4,5,7) padding1=0
        %fidx_327 = arith.constant 327 : index
        %bidx_327 = arith.constant 578 : index
        %smidx_327 = arith.constant 998 : index
        %ssidx_327 = arith.constant 1418 : index
        %bias_327 = memref.load %in_v[%bidx_327] : memref<1511xi32>
        %smean_327 = memref.load %in_v[%smidx_327] : memref<1511xi32>
        %sscale_327 = memref.load %in_v[%ssidx_327] : memref<1511xi32>
        %dilation_327 = arith.constant 21 : i32
        %padding_327 = arith.constant 84 : i32
        %ki0_327 = arith.constant 4 : i32
        %ki1_327 = arith.constant 5 : i32
        %ki2_327 = arith.constant 7 : i32
        %pad1_327 = arith.constant 0 : i32
        %outidx_327 = arith.constant 327 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_327, %smean_327, %sscale_327, %dilation_327, %padding_327, %ki0_327, %ki1_327, %ki2_327, %pad1_327, %out_v, %outidx_327) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 328: dilation=21 padding=84 kernel=(4,5,8) padding1=1
        %fidx_328 = arith.constant 328 : index
        %bidx_328 = arith.constant 579 : index
        %smidx_328 = arith.constant 999 : index
        %ssidx_328 = arith.constant 1419 : index
        %bias_328 = memref.load %in_v[%bidx_328] : memref<1511xi32>
        %smean_328 = memref.load %in_v[%smidx_328] : memref<1511xi32>
        %sscale_328 = memref.load %in_v[%ssidx_328] : memref<1511xi32>
        %dilation_328 = arith.constant 21 : i32
        %padding_328 = arith.constant 84 : i32
        %ki0_328 = arith.constant 4 : i32
        %ki1_328 = arith.constant 5 : i32
        %ki2_328 = arith.constant 8 : i32
        %pad1_328 = arith.constant 1 : i32
        %outidx_328 = arith.constant 328 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_328, %smean_328, %sscale_328, %dilation_328, %padding_328, %ki0_328, %ki1_328, %ki2_328, %pad1_328, %out_v, %outidx_328) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 329: dilation=21 padding=84 kernel=(4,6,7) padding1=0
        %fidx_329 = arith.constant 329 : index
        %bidx_329 = arith.constant 580 : index
        %smidx_329 = arith.constant 1000 : index
        %ssidx_329 = arith.constant 1420 : index
        %bias_329 = memref.load %in_v[%bidx_329] : memref<1511xi32>
        %smean_329 = memref.load %in_v[%smidx_329] : memref<1511xi32>
        %sscale_329 = memref.load %in_v[%ssidx_329] : memref<1511xi32>
        %dilation_329 = arith.constant 21 : i32
        %padding_329 = arith.constant 84 : i32
        %ki0_329 = arith.constant 4 : i32
        %ki1_329 = arith.constant 6 : i32
        %ki2_329 = arith.constant 7 : i32
        %pad1_329 = arith.constant 0 : i32
        %outidx_329 = arith.constant 329 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_329, %smean_329, %sscale_329, %dilation_329, %padding_329, %ki0_329, %ki1_329, %ki2_329, %pad1_329, %out_v, %outidx_329) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 330: dilation=21 padding=84 kernel=(4,6,8) padding1=1
        %fidx_330 = arith.constant 330 : index
        %bidx_330 = arith.constant 581 : index
        %smidx_330 = arith.constant 1001 : index
        %ssidx_330 = arith.constant 1421 : index
        %bias_330 = memref.load %in_v[%bidx_330] : memref<1511xi32>
        %smean_330 = memref.load %in_v[%smidx_330] : memref<1511xi32>
        %sscale_330 = memref.load %in_v[%ssidx_330] : memref<1511xi32>
        %dilation_330 = arith.constant 21 : i32
        %padding_330 = arith.constant 84 : i32
        %ki0_330 = arith.constant 4 : i32
        %ki1_330 = arith.constant 6 : i32
        %ki2_330 = arith.constant 8 : i32
        %pad1_330 = arith.constant 1 : i32
        %outidx_330 = arith.constant 330 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_330, %smean_330, %sscale_330, %dilation_330, %padding_330, %ki0_330, %ki1_330, %ki2_330, %pad1_330, %out_v, %outidx_330) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 331: dilation=21 padding=84 kernel=(4,7,8) padding1=0
        %fidx_331 = arith.constant 331 : index
        %bidx_331 = arith.constant 582 : index
        %smidx_331 = arith.constant 1002 : index
        %ssidx_331 = arith.constant 1422 : index
        %bias_331 = memref.load %in_v[%bidx_331] : memref<1511xi32>
        %smean_331 = memref.load %in_v[%smidx_331] : memref<1511xi32>
        %sscale_331 = memref.load %in_v[%ssidx_331] : memref<1511xi32>
        %dilation_331 = arith.constant 21 : i32
        %padding_331 = arith.constant 84 : i32
        %ki0_331 = arith.constant 4 : i32
        %ki1_331 = arith.constant 7 : i32
        %ki2_331 = arith.constant 8 : i32
        %pad1_331 = arith.constant 0 : i32
        %outidx_331 = arith.constant 331 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_331, %smean_331, %sscale_331, %dilation_331, %padding_331, %ki0_331, %ki1_331, %ki2_331, %pad1_331, %out_v, %outidx_331) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 332: dilation=21 padding=84 kernel=(5,6,7) padding1=1
        %fidx_332 = arith.constant 332 : index
        %bidx_332 = arith.constant 583 : index
        %smidx_332 = arith.constant 1003 : index
        %ssidx_332 = arith.constant 1423 : index
        %bias_332 = memref.load %in_v[%bidx_332] : memref<1511xi32>
        %smean_332 = memref.load %in_v[%smidx_332] : memref<1511xi32>
        %sscale_332 = memref.load %in_v[%ssidx_332] : memref<1511xi32>
        %dilation_332 = arith.constant 21 : i32
        %padding_332 = arith.constant 84 : i32
        %ki0_332 = arith.constant 5 : i32
        %ki1_332 = arith.constant 6 : i32
        %ki2_332 = arith.constant 7 : i32
        %pad1_332 = arith.constant 1 : i32
        %outidx_332 = arith.constant 332 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_332, %smean_332, %sscale_332, %dilation_332, %padding_332, %ki0_332, %ki1_332, %ki2_332, %pad1_332, %out_v, %outidx_332) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 333: dilation=21 padding=84 kernel=(5,6,8) padding1=0
        %fidx_333 = arith.constant 333 : index
        %bidx_333 = arith.constant 584 : index
        %smidx_333 = arith.constant 1004 : index
        %ssidx_333 = arith.constant 1424 : index
        %bias_333 = memref.load %in_v[%bidx_333] : memref<1511xi32>
        %smean_333 = memref.load %in_v[%smidx_333] : memref<1511xi32>
        %sscale_333 = memref.load %in_v[%ssidx_333] : memref<1511xi32>
        %dilation_333 = arith.constant 21 : i32
        %padding_333 = arith.constant 84 : i32
        %ki0_333 = arith.constant 5 : i32
        %ki1_333 = arith.constant 6 : i32
        %ki2_333 = arith.constant 8 : i32
        %pad1_333 = arith.constant 0 : i32
        %outidx_333 = arith.constant 333 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_333, %smean_333, %sscale_333, %dilation_333, %padding_333, %ki0_333, %ki1_333, %ki2_333, %pad1_333, %out_v, %outidx_333) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 334: dilation=21 padding=84 kernel=(5,7,8) padding1=1
        %fidx_334 = arith.constant 334 : index
        %bidx_334 = arith.constant 585 : index
        %smidx_334 = arith.constant 1005 : index
        %ssidx_334 = arith.constant 1425 : index
        %bias_334 = memref.load %in_v[%bidx_334] : memref<1511xi32>
        %smean_334 = memref.load %in_v[%smidx_334] : memref<1511xi32>
        %sscale_334 = memref.load %in_v[%ssidx_334] : memref<1511xi32>
        %dilation_334 = arith.constant 21 : i32
        %padding_334 = arith.constant 84 : i32
        %ki0_334 = arith.constant 5 : i32
        %ki1_334 = arith.constant 7 : i32
        %ki2_334 = arith.constant 8 : i32
        %pad1_334 = arith.constant 1 : i32
        %outidx_334 = arith.constant 334 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_334, %smean_334, %sscale_334, %dilation_334, %padding_334, %ki0_334, %ki1_334, %ki2_334, %pad1_334, %out_v, %outidx_334) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 335: dilation=21 padding=84 kernel=(6,7,8) padding1=0
        %fidx_335 = arith.constant 335 : index
        %bidx_335 = arith.constant 586 : index
        %smidx_335 = arith.constant 1006 : index
        %ssidx_335 = arith.constant 1426 : index
        %bias_335 = memref.load %in_v[%bidx_335] : memref<1511xi32>
        %smean_335 = memref.load %in_v[%smidx_335] : memref<1511xi32>
        %sscale_335 = memref.load %in_v[%ssidx_335] : memref<1511xi32>
        %dilation_335 = arith.constant 21 : i32
        %padding_335 = arith.constant 84 : i32
        %ki0_335 = arith.constant 6 : i32
        %ki1_335 = arith.constant 7 : i32
        %ki2_335 = arith.constant 8 : i32
        %pad1_335 = arith.constant 0 : i32
        %outidx_335 = arith.constant 335 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_335, %smean_335, %sscale_335, %dilation_335, %padding_335, %ki0_335, %ki1_335, %ki2_335, %pad1_335, %out_v, %outidx_335) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 336: dilation=31 padding=124 kernel=(0,1,2) padding1=0
        %fidx_336 = arith.constant 336 : index
        %bidx_336 = arith.constant 587 : index
        %smidx_336 = arith.constant 1007 : index
        %ssidx_336 = arith.constant 1427 : index
        %bias_336 = memref.load %in_v[%bidx_336] : memref<1511xi32>
        %smean_336 = memref.load %in_v[%smidx_336] : memref<1511xi32>
        %sscale_336 = memref.load %in_v[%ssidx_336] : memref<1511xi32>
        %dilation_336 = arith.constant 31 : i32
        %padding_336 = arith.constant 124 : i32
        %ki0_336 = arith.constant 0 : i32
        %ki1_336 = arith.constant 1 : i32
        %ki2_336 = arith.constant 2 : i32
        %pad1_336 = arith.constant 0 : i32
        %outidx_336 = arith.constant 336 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_336, %smean_336, %sscale_336, %dilation_336, %padding_336, %ki0_336, %ki1_336, %ki2_336, %pad1_336, %out_v, %outidx_336) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 337: dilation=31 padding=124 kernel=(0,1,3) padding1=1
        %fidx_337 = arith.constant 337 : index
        %bidx_337 = arith.constant 588 : index
        %smidx_337 = arith.constant 1008 : index
        %ssidx_337 = arith.constant 1428 : index
        %bias_337 = memref.load %in_v[%bidx_337] : memref<1511xi32>
        %smean_337 = memref.load %in_v[%smidx_337] : memref<1511xi32>
        %sscale_337 = memref.load %in_v[%ssidx_337] : memref<1511xi32>
        %dilation_337 = arith.constant 31 : i32
        %padding_337 = arith.constant 124 : i32
        %ki0_337 = arith.constant 0 : i32
        %ki1_337 = arith.constant 1 : i32
        %ki2_337 = arith.constant 3 : i32
        %pad1_337 = arith.constant 1 : i32
        %outidx_337 = arith.constant 337 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_337, %smean_337, %sscale_337, %dilation_337, %padding_337, %ki0_337, %ki1_337, %ki2_337, %pad1_337, %out_v, %outidx_337) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 338: dilation=31 padding=124 kernel=(0,1,4) padding1=0
        %fidx_338 = arith.constant 338 : index
        %bidx_338 = arith.constant 589 : index
        %smidx_338 = arith.constant 1009 : index
        %ssidx_338 = arith.constant 1429 : index
        %bias_338 = memref.load %in_v[%bidx_338] : memref<1511xi32>
        %smean_338 = memref.load %in_v[%smidx_338] : memref<1511xi32>
        %sscale_338 = memref.load %in_v[%ssidx_338] : memref<1511xi32>
        %dilation_338 = arith.constant 31 : i32
        %padding_338 = arith.constant 124 : i32
        %ki0_338 = arith.constant 0 : i32
        %ki1_338 = arith.constant 1 : i32
        %ki2_338 = arith.constant 4 : i32
        %pad1_338 = arith.constant 0 : i32
        %outidx_338 = arith.constant 338 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_338, %smean_338, %sscale_338, %dilation_338, %padding_338, %ki0_338, %ki1_338, %ki2_338, %pad1_338, %out_v, %outidx_338) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 339: dilation=31 padding=124 kernel=(0,1,5) padding1=1
        %fidx_339 = arith.constant 339 : index
        %bidx_339 = arith.constant 590 : index
        %smidx_339 = arith.constant 1010 : index
        %ssidx_339 = arith.constant 1430 : index
        %bias_339 = memref.load %in_v[%bidx_339] : memref<1511xi32>
        %smean_339 = memref.load %in_v[%smidx_339] : memref<1511xi32>
        %sscale_339 = memref.load %in_v[%ssidx_339] : memref<1511xi32>
        %dilation_339 = arith.constant 31 : i32
        %padding_339 = arith.constant 124 : i32
        %ki0_339 = arith.constant 0 : i32
        %ki1_339 = arith.constant 1 : i32
        %ki2_339 = arith.constant 5 : i32
        %pad1_339 = arith.constant 1 : i32
        %outidx_339 = arith.constant 339 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_339, %smean_339, %sscale_339, %dilation_339, %padding_339, %ki0_339, %ki1_339, %ki2_339, %pad1_339, %out_v, %outidx_339) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 340: dilation=31 padding=124 kernel=(0,1,6) padding1=0
        %fidx_340 = arith.constant 340 : index
        %bidx_340 = arith.constant 591 : index
        %smidx_340 = arith.constant 1011 : index
        %ssidx_340 = arith.constant 1431 : index
        %bias_340 = memref.load %in_v[%bidx_340] : memref<1511xi32>
        %smean_340 = memref.load %in_v[%smidx_340] : memref<1511xi32>
        %sscale_340 = memref.load %in_v[%ssidx_340] : memref<1511xi32>
        %dilation_340 = arith.constant 31 : i32
        %padding_340 = arith.constant 124 : i32
        %ki0_340 = arith.constant 0 : i32
        %ki1_340 = arith.constant 1 : i32
        %ki2_340 = arith.constant 6 : i32
        %pad1_340 = arith.constant 0 : i32
        %outidx_340 = arith.constant 340 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_340, %smean_340, %sscale_340, %dilation_340, %padding_340, %ki0_340, %ki1_340, %ki2_340, %pad1_340, %out_v, %outidx_340) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 341: dilation=31 padding=124 kernel=(0,1,7) padding1=1
        %fidx_341 = arith.constant 341 : index
        %bidx_341 = arith.constant 592 : index
        %smidx_341 = arith.constant 1012 : index
        %ssidx_341 = arith.constant 1432 : index
        %bias_341 = memref.load %in_v[%bidx_341] : memref<1511xi32>
        %smean_341 = memref.load %in_v[%smidx_341] : memref<1511xi32>
        %sscale_341 = memref.load %in_v[%ssidx_341] : memref<1511xi32>
        %dilation_341 = arith.constant 31 : i32
        %padding_341 = arith.constant 124 : i32
        %ki0_341 = arith.constant 0 : i32
        %ki1_341 = arith.constant 1 : i32
        %ki2_341 = arith.constant 7 : i32
        %pad1_341 = arith.constant 1 : i32
        %outidx_341 = arith.constant 341 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_341, %smean_341, %sscale_341, %dilation_341, %padding_341, %ki0_341, %ki1_341, %ki2_341, %pad1_341, %out_v, %outidx_341) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 342: dilation=31 padding=124 kernel=(0,1,8) padding1=0
        %fidx_342 = arith.constant 342 : index
        %bidx_342 = arith.constant 593 : index
        %smidx_342 = arith.constant 1013 : index
        %ssidx_342 = arith.constant 1433 : index
        %bias_342 = memref.load %in_v[%bidx_342] : memref<1511xi32>
        %smean_342 = memref.load %in_v[%smidx_342] : memref<1511xi32>
        %sscale_342 = memref.load %in_v[%ssidx_342] : memref<1511xi32>
        %dilation_342 = arith.constant 31 : i32
        %padding_342 = arith.constant 124 : i32
        %ki0_342 = arith.constant 0 : i32
        %ki1_342 = arith.constant 1 : i32
        %ki2_342 = arith.constant 8 : i32
        %pad1_342 = arith.constant 0 : i32
        %outidx_342 = arith.constant 342 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_342, %smean_342, %sscale_342, %dilation_342, %padding_342, %ki0_342, %ki1_342, %ki2_342, %pad1_342, %out_v, %outidx_342) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 343: dilation=31 padding=124 kernel=(0,2,3) padding1=1
        %fidx_343 = arith.constant 343 : index
        %bidx_343 = arith.constant 594 : index
        %smidx_343 = arith.constant 1014 : index
        %ssidx_343 = arith.constant 1434 : index
        %bias_343 = memref.load %in_v[%bidx_343] : memref<1511xi32>
        %smean_343 = memref.load %in_v[%smidx_343] : memref<1511xi32>
        %sscale_343 = memref.load %in_v[%ssidx_343] : memref<1511xi32>
        %dilation_343 = arith.constant 31 : i32
        %padding_343 = arith.constant 124 : i32
        %ki0_343 = arith.constant 0 : i32
        %ki1_343 = arith.constant 2 : i32
        %ki2_343 = arith.constant 3 : i32
        %pad1_343 = arith.constant 1 : i32
        %outidx_343 = arith.constant 343 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_343, %smean_343, %sscale_343, %dilation_343, %padding_343, %ki0_343, %ki1_343, %ki2_343, %pad1_343, %out_v, %outidx_343) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 344: dilation=31 padding=124 kernel=(0,2,4) padding1=0
        %fidx_344 = arith.constant 344 : index
        %bidx_344 = arith.constant 595 : index
        %smidx_344 = arith.constant 1015 : index
        %ssidx_344 = arith.constant 1435 : index
        %bias_344 = memref.load %in_v[%bidx_344] : memref<1511xi32>
        %smean_344 = memref.load %in_v[%smidx_344] : memref<1511xi32>
        %sscale_344 = memref.load %in_v[%ssidx_344] : memref<1511xi32>
        %dilation_344 = arith.constant 31 : i32
        %padding_344 = arith.constant 124 : i32
        %ki0_344 = arith.constant 0 : i32
        %ki1_344 = arith.constant 2 : i32
        %ki2_344 = arith.constant 4 : i32
        %pad1_344 = arith.constant 0 : i32
        %outidx_344 = arith.constant 344 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_344, %smean_344, %sscale_344, %dilation_344, %padding_344, %ki0_344, %ki1_344, %ki2_344, %pad1_344, %out_v, %outidx_344) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 345: dilation=31 padding=124 kernel=(0,2,5) padding1=1
        %fidx_345 = arith.constant 345 : index
        %bidx_345 = arith.constant 596 : index
        %smidx_345 = arith.constant 1016 : index
        %ssidx_345 = arith.constant 1436 : index
        %bias_345 = memref.load %in_v[%bidx_345] : memref<1511xi32>
        %smean_345 = memref.load %in_v[%smidx_345] : memref<1511xi32>
        %sscale_345 = memref.load %in_v[%ssidx_345] : memref<1511xi32>
        %dilation_345 = arith.constant 31 : i32
        %padding_345 = arith.constant 124 : i32
        %ki0_345 = arith.constant 0 : i32
        %ki1_345 = arith.constant 2 : i32
        %ki2_345 = arith.constant 5 : i32
        %pad1_345 = arith.constant 1 : i32
        %outidx_345 = arith.constant 345 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_345, %smean_345, %sscale_345, %dilation_345, %padding_345, %ki0_345, %ki1_345, %ki2_345, %pad1_345, %out_v, %outidx_345) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 346: dilation=31 padding=124 kernel=(0,2,6) padding1=0
        %fidx_346 = arith.constant 346 : index
        %bidx_346 = arith.constant 597 : index
        %smidx_346 = arith.constant 1017 : index
        %ssidx_346 = arith.constant 1437 : index
        %bias_346 = memref.load %in_v[%bidx_346] : memref<1511xi32>
        %smean_346 = memref.load %in_v[%smidx_346] : memref<1511xi32>
        %sscale_346 = memref.load %in_v[%ssidx_346] : memref<1511xi32>
        %dilation_346 = arith.constant 31 : i32
        %padding_346 = arith.constant 124 : i32
        %ki0_346 = arith.constant 0 : i32
        %ki1_346 = arith.constant 2 : i32
        %ki2_346 = arith.constant 6 : i32
        %pad1_346 = arith.constant 0 : i32
        %outidx_346 = arith.constant 346 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_346, %smean_346, %sscale_346, %dilation_346, %padding_346, %ki0_346, %ki1_346, %ki2_346, %pad1_346, %out_v, %outidx_346) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 347: dilation=31 padding=124 kernel=(0,2,7) padding1=1
        %fidx_347 = arith.constant 347 : index
        %bidx_347 = arith.constant 598 : index
        %smidx_347 = arith.constant 1018 : index
        %ssidx_347 = arith.constant 1438 : index
        %bias_347 = memref.load %in_v[%bidx_347] : memref<1511xi32>
        %smean_347 = memref.load %in_v[%smidx_347] : memref<1511xi32>
        %sscale_347 = memref.load %in_v[%ssidx_347] : memref<1511xi32>
        %dilation_347 = arith.constant 31 : i32
        %padding_347 = arith.constant 124 : i32
        %ki0_347 = arith.constant 0 : i32
        %ki1_347 = arith.constant 2 : i32
        %ki2_347 = arith.constant 7 : i32
        %pad1_347 = arith.constant 1 : i32
        %outidx_347 = arith.constant 347 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_347, %smean_347, %sscale_347, %dilation_347, %padding_347, %ki0_347, %ki1_347, %ki2_347, %pad1_347, %out_v, %outidx_347) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 348: dilation=31 padding=124 kernel=(0,2,8) padding1=0
        %fidx_348 = arith.constant 348 : index
        %bidx_348 = arith.constant 599 : index
        %smidx_348 = arith.constant 1019 : index
        %ssidx_348 = arith.constant 1439 : index
        %bias_348 = memref.load %in_v[%bidx_348] : memref<1511xi32>
        %smean_348 = memref.load %in_v[%smidx_348] : memref<1511xi32>
        %sscale_348 = memref.load %in_v[%ssidx_348] : memref<1511xi32>
        %dilation_348 = arith.constant 31 : i32
        %padding_348 = arith.constant 124 : i32
        %ki0_348 = arith.constant 0 : i32
        %ki1_348 = arith.constant 2 : i32
        %ki2_348 = arith.constant 8 : i32
        %pad1_348 = arith.constant 0 : i32
        %outidx_348 = arith.constant 348 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_348, %smean_348, %sscale_348, %dilation_348, %padding_348, %ki0_348, %ki1_348, %ki2_348, %pad1_348, %out_v, %outidx_348) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 349: dilation=31 padding=124 kernel=(0,3,4) padding1=1
        %fidx_349 = arith.constant 349 : index
        %bidx_349 = arith.constant 600 : index
        %smidx_349 = arith.constant 1020 : index
        %ssidx_349 = arith.constant 1440 : index
        %bias_349 = memref.load %in_v[%bidx_349] : memref<1511xi32>
        %smean_349 = memref.load %in_v[%smidx_349] : memref<1511xi32>
        %sscale_349 = memref.load %in_v[%ssidx_349] : memref<1511xi32>
        %dilation_349 = arith.constant 31 : i32
        %padding_349 = arith.constant 124 : i32
        %ki0_349 = arith.constant 0 : i32
        %ki1_349 = arith.constant 3 : i32
        %ki2_349 = arith.constant 4 : i32
        %pad1_349 = arith.constant 1 : i32
        %outidx_349 = arith.constant 349 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_349, %smean_349, %sscale_349, %dilation_349, %padding_349, %ki0_349, %ki1_349, %ki2_349, %pad1_349, %out_v, %outidx_349) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 350: dilation=31 padding=124 kernel=(0,3,5) padding1=0
        %fidx_350 = arith.constant 350 : index
        %bidx_350 = arith.constant 601 : index
        %smidx_350 = arith.constant 1021 : index
        %ssidx_350 = arith.constant 1441 : index
        %bias_350 = memref.load %in_v[%bidx_350] : memref<1511xi32>
        %smean_350 = memref.load %in_v[%smidx_350] : memref<1511xi32>
        %sscale_350 = memref.load %in_v[%ssidx_350] : memref<1511xi32>
        %dilation_350 = arith.constant 31 : i32
        %padding_350 = arith.constant 124 : i32
        %ki0_350 = arith.constant 0 : i32
        %ki1_350 = arith.constant 3 : i32
        %ki2_350 = arith.constant 5 : i32
        %pad1_350 = arith.constant 0 : i32
        %outidx_350 = arith.constant 350 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_350, %smean_350, %sscale_350, %dilation_350, %padding_350, %ki0_350, %ki1_350, %ki2_350, %pad1_350, %out_v, %outidx_350) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 351: dilation=31 padding=124 kernel=(0,3,6) padding1=1
        %fidx_351 = arith.constant 351 : index
        %bidx_351 = arith.constant 602 : index
        %smidx_351 = arith.constant 1022 : index
        %ssidx_351 = arith.constant 1442 : index
        %bias_351 = memref.load %in_v[%bidx_351] : memref<1511xi32>
        %smean_351 = memref.load %in_v[%smidx_351] : memref<1511xi32>
        %sscale_351 = memref.load %in_v[%ssidx_351] : memref<1511xi32>
        %dilation_351 = arith.constant 31 : i32
        %padding_351 = arith.constant 124 : i32
        %ki0_351 = arith.constant 0 : i32
        %ki1_351 = arith.constant 3 : i32
        %ki2_351 = arith.constant 6 : i32
        %pad1_351 = arith.constant 1 : i32
        %outidx_351 = arith.constant 351 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_351, %smean_351, %sscale_351, %dilation_351, %padding_351, %ki0_351, %ki1_351, %ki2_351, %pad1_351, %out_v, %outidx_351) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 352: dilation=31 padding=124 kernel=(0,3,7) padding1=0
        %fidx_352 = arith.constant 352 : index
        %bidx_352 = arith.constant 603 : index
        %smidx_352 = arith.constant 1023 : index
        %ssidx_352 = arith.constant 1443 : index
        %bias_352 = memref.load %in_v[%bidx_352] : memref<1511xi32>
        %smean_352 = memref.load %in_v[%smidx_352] : memref<1511xi32>
        %sscale_352 = memref.load %in_v[%ssidx_352] : memref<1511xi32>
        %dilation_352 = arith.constant 31 : i32
        %padding_352 = arith.constant 124 : i32
        %ki0_352 = arith.constant 0 : i32
        %ki1_352 = arith.constant 3 : i32
        %ki2_352 = arith.constant 7 : i32
        %pad1_352 = arith.constant 0 : i32
        %outidx_352 = arith.constant 352 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_352, %smean_352, %sscale_352, %dilation_352, %padding_352, %ki0_352, %ki1_352, %ki2_352, %pad1_352, %out_v, %outidx_352) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 353: dilation=31 padding=124 kernel=(0,3,8) padding1=1
        %fidx_353 = arith.constant 353 : index
        %bidx_353 = arith.constant 604 : index
        %smidx_353 = arith.constant 1024 : index
        %ssidx_353 = arith.constant 1444 : index
        %bias_353 = memref.load %in_v[%bidx_353] : memref<1511xi32>
        %smean_353 = memref.load %in_v[%smidx_353] : memref<1511xi32>
        %sscale_353 = memref.load %in_v[%ssidx_353] : memref<1511xi32>
        %dilation_353 = arith.constant 31 : i32
        %padding_353 = arith.constant 124 : i32
        %ki0_353 = arith.constant 0 : i32
        %ki1_353 = arith.constant 3 : i32
        %ki2_353 = arith.constant 8 : i32
        %pad1_353 = arith.constant 1 : i32
        %outidx_353 = arith.constant 353 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_353, %smean_353, %sscale_353, %dilation_353, %padding_353, %ki0_353, %ki1_353, %ki2_353, %pad1_353, %out_v, %outidx_353) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 354: dilation=31 padding=124 kernel=(0,4,5) padding1=0
        %fidx_354 = arith.constant 354 : index
        %bidx_354 = arith.constant 605 : index
        %smidx_354 = arith.constant 1025 : index
        %ssidx_354 = arith.constant 1445 : index
        %bias_354 = memref.load %in_v[%bidx_354] : memref<1511xi32>
        %smean_354 = memref.load %in_v[%smidx_354] : memref<1511xi32>
        %sscale_354 = memref.load %in_v[%ssidx_354] : memref<1511xi32>
        %dilation_354 = arith.constant 31 : i32
        %padding_354 = arith.constant 124 : i32
        %ki0_354 = arith.constant 0 : i32
        %ki1_354 = arith.constant 4 : i32
        %ki2_354 = arith.constant 5 : i32
        %pad1_354 = arith.constant 0 : i32
        %outidx_354 = arith.constant 354 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_354, %smean_354, %sscale_354, %dilation_354, %padding_354, %ki0_354, %ki1_354, %ki2_354, %pad1_354, %out_v, %outidx_354) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 355: dilation=31 padding=124 kernel=(0,4,6) padding1=1
        %fidx_355 = arith.constant 355 : index
        %bidx_355 = arith.constant 606 : index
        %smidx_355 = arith.constant 1026 : index
        %ssidx_355 = arith.constant 1446 : index
        %bias_355 = memref.load %in_v[%bidx_355] : memref<1511xi32>
        %smean_355 = memref.load %in_v[%smidx_355] : memref<1511xi32>
        %sscale_355 = memref.load %in_v[%ssidx_355] : memref<1511xi32>
        %dilation_355 = arith.constant 31 : i32
        %padding_355 = arith.constant 124 : i32
        %ki0_355 = arith.constant 0 : i32
        %ki1_355 = arith.constant 4 : i32
        %ki2_355 = arith.constant 6 : i32
        %pad1_355 = arith.constant 1 : i32
        %outidx_355 = arith.constant 355 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_355, %smean_355, %sscale_355, %dilation_355, %padding_355, %ki0_355, %ki1_355, %ki2_355, %pad1_355, %out_v, %outidx_355) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 356: dilation=31 padding=124 kernel=(0,4,7) padding1=0
        %fidx_356 = arith.constant 356 : index
        %bidx_356 = arith.constant 607 : index
        %smidx_356 = arith.constant 1027 : index
        %ssidx_356 = arith.constant 1447 : index
        %bias_356 = memref.load %in_v[%bidx_356] : memref<1511xi32>
        %smean_356 = memref.load %in_v[%smidx_356] : memref<1511xi32>
        %sscale_356 = memref.load %in_v[%ssidx_356] : memref<1511xi32>
        %dilation_356 = arith.constant 31 : i32
        %padding_356 = arith.constant 124 : i32
        %ki0_356 = arith.constant 0 : i32
        %ki1_356 = arith.constant 4 : i32
        %ki2_356 = arith.constant 7 : i32
        %pad1_356 = arith.constant 0 : i32
        %outidx_356 = arith.constant 356 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_356, %smean_356, %sscale_356, %dilation_356, %padding_356, %ki0_356, %ki1_356, %ki2_356, %pad1_356, %out_v, %outidx_356) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 357: dilation=31 padding=124 kernel=(0,4,8) padding1=1
        %fidx_357 = arith.constant 357 : index
        %bidx_357 = arith.constant 608 : index
        %smidx_357 = arith.constant 1028 : index
        %ssidx_357 = arith.constant 1448 : index
        %bias_357 = memref.load %in_v[%bidx_357] : memref<1511xi32>
        %smean_357 = memref.load %in_v[%smidx_357] : memref<1511xi32>
        %sscale_357 = memref.load %in_v[%ssidx_357] : memref<1511xi32>
        %dilation_357 = arith.constant 31 : i32
        %padding_357 = arith.constant 124 : i32
        %ki0_357 = arith.constant 0 : i32
        %ki1_357 = arith.constant 4 : i32
        %ki2_357 = arith.constant 8 : i32
        %pad1_357 = arith.constant 1 : i32
        %outidx_357 = arith.constant 357 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_357, %smean_357, %sscale_357, %dilation_357, %padding_357, %ki0_357, %ki1_357, %ki2_357, %pad1_357, %out_v, %outidx_357) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 358: dilation=31 padding=124 kernel=(0,5,6) padding1=0
        %fidx_358 = arith.constant 358 : index
        %bidx_358 = arith.constant 609 : index
        %smidx_358 = arith.constant 1029 : index
        %ssidx_358 = arith.constant 1449 : index
        %bias_358 = memref.load %in_v[%bidx_358] : memref<1511xi32>
        %smean_358 = memref.load %in_v[%smidx_358] : memref<1511xi32>
        %sscale_358 = memref.load %in_v[%ssidx_358] : memref<1511xi32>
        %dilation_358 = arith.constant 31 : i32
        %padding_358 = arith.constant 124 : i32
        %ki0_358 = arith.constant 0 : i32
        %ki1_358 = arith.constant 5 : i32
        %ki2_358 = arith.constant 6 : i32
        %pad1_358 = arith.constant 0 : i32
        %outidx_358 = arith.constant 358 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_358, %smean_358, %sscale_358, %dilation_358, %padding_358, %ki0_358, %ki1_358, %ki2_358, %pad1_358, %out_v, %outidx_358) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 359: dilation=31 padding=124 kernel=(0,5,7) padding1=1
        %fidx_359 = arith.constant 359 : index
        %bidx_359 = arith.constant 610 : index
        %smidx_359 = arith.constant 1030 : index
        %ssidx_359 = arith.constant 1450 : index
        %bias_359 = memref.load %in_v[%bidx_359] : memref<1511xi32>
        %smean_359 = memref.load %in_v[%smidx_359] : memref<1511xi32>
        %sscale_359 = memref.load %in_v[%ssidx_359] : memref<1511xi32>
        %dilation_359 = arith.constant 31 : i32
        %padding_359 = arith.constant 124 : i32
        %ki0_359 = arith.constant 0 : i32
        %ki1_359 = arith.constant 5 : i32
        %ki2_359 = arith.constant 7 : i32
        %pad1_359 = arith.constant 1 : i32
        %outidx_359 = arith.constant 359 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_359, %smean_359, %sscale_359, %dilation_359, %padding_359, %ki0_359, %ki1_359, %ki2_359, %pad1_359, %out_v, %outidx_359) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 360: dilation=31 padding=124 kernel=(0,5,8) padding1=0
        %fidx_360 = arith.constant 360 : index
        %bidx_360 = arith.constant 611 : index
        %smidx_360 = arith.constant 1031 : index
        %ssidx_360 = arith.constant 1451 : index
        %bias_360 = memref.load %in_v[%bidx_360] : memref<1511xi32>
        %smean_360 = memref.load %in_v[%smidx_360] : memref<1511xi32>
        %sscale_360 = memref.load %in_v[%ssidx_360] : memref<1511xi32>
        %dilation_360 = arith.constant 31 : i32
        %padding_360 = arith.constant 124 : i32
        %ki0_360 = arith.constant 0 : i32
        %ki1_360 = arith.constant 5 : i32
        %ki2_360 = arith.constant 8 : i32
        %pad1_360 = arith.constant 0 : i32
        %outidx_360 = arith.constant 360 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_360, %smean_360, %sscale_360, %dilation_360, %padding_360, %ki0_360, %ki1_360, %ki2_360, %pad1_360, %out_v, %outidx_360) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 361: dilation=31 padding=124 kernel=(0,6,7) padding1=1
        %fidx_361 = arith.constant 361 : index
        %bidx_361 = arith.constant 612 : index
        %smidx_361 = arith.constant 1032 : index
        %ssidx_361 = arith.constant 1452 : index
        %bias_361 = memref.load %in_v[%bidx_361] : memref<1511xi32>
        %smean_361 = memref.load %in_v[%smidx_361] : memref<1511xi32>
        %sscale_361 = memref.load %in_v[%ssidx_361] : memref<1511xi32>
        %dilation_361 = arith.constant 31 : i32
        %padding_361 = arith.constant 124 : i32
        %ki0_361 = arith.constant 0 : i32
        %ki1_361 = arith.constant 6 : i32
        %ki2_361 = arith.constant 7 : i32
        %pad1_361 = arith.constant 1 : i32
        %outidx_361 = arith.constant 361 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_361, %smean_361, %sscale_361, %dilation_361, %padding_361, %ki0_361, %ki1_361, %ki2_361, %pad1_361, %out_v, %outidx_361) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 362: dilation=31 padding=124 kernel=(0,6,8) padding1=0
        %fidx_362 = arith.constant 362 : index
        %bidx_362 = arith.constant 613 : index
        %smidx_362 = arith.constant 1033 : index
        %ssidx_362 = arith.constant 1453 : index
        %bias_362 = memref.load %in_v[%bidx_362] : memref<1511xi32>
        %smean_362 = memref.load %in_v[%smidx_362] : memref<1511xi32>
        %sscale_362 = memref.load %in_v[%ssidx_362] : memref<1511xi32>
        %dilation_362 = arith.constant 31 : i32
        %padding_362 = arith.constant 124 : i32
        %ki0_362 = arith.constant 0 : i32
        %ki1_362 = arith.constant 6 : i32
        %ki2_362 = arith.constant 8 : i32
        %pad1_362 = arith.constant 0 : i32
        %outidx_362 = arith.constant 362 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_362, %smean_362, %sscale_362, %dilation_362, %padding_362, %ki0_362, %ki1_362, %ki2_362, %pad1_362, %out_v, %outidx_362) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 363: dilation=31 padding=124 kernel=(0,7,8) padding1=1
        %fidx_363 = arith.constant 363 : index
        %bidx_363 = arith.constant 614 : index
        %smidx_363 = arith.constant 1034 : index
        %ssidx_363 = arith.constant 1454 : index
        %bias_363 = memref.load %in_v[%bidx_363] : memref<1511xi32>
        %smean_363 = memref.load %in_v[%smidx_363] : memref<1511xi32>
        %sscale_363 = memref.load %in_v[%ssidx_363] : memref<1511xi32>
        %dilation_363 = arith.constant 31 : i32
        %padding_363 = arith.constant 124 : i32
        %ki0_363 = arith.constant 0 : i32
        %ki1_363 = arith.constant 7 : i32
        %ki2_363 = arith.constant 8 : i32
        %pad1_363 = arith.constant 1 : i32
        %outidx_363 = arith.constant 363 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_363, %smean_363, %sscale_363, %dilation_363, %padding_363, %ki0_363, %ki1_363, %ki2_363, %pad1_363, %out_v, %outidx_363) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 364: dilation=31 padding=124 kernel=(1,2,3) padding1=0
        %fidx_364 = arith.constant 364 : index
        %bidx_364 = arith.constant 615 : index
        %smidx_364 = arith.constant 1035 : index
        %ssidx_364 = arith.constant 1455 : index
        %bias_364 = memref.load %in_v[%bidx_364] : memref<1511xi32>
        %smean_364 = memref.load %in_v[%smidx_364] : memref<1511xi32>
        %sscale_364 = memref.load %in_v[%ssidx_364] : memref<1511xi32>
        %dilation_364 = arith.constant 31 : i32
        %padding_364 = arith.constant 124 : i32
        %ki0_364 = arith.constant 1 : i32
        %ki1_364 = arith.constant 2 : i32
        %ki2_364 = arith.constant 3 : i32
        %pad1_364 = arith.constant 0 : i32
        %outidx_364 = arith.constant 364 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_364, %smean_364, %sscale_364, %dilation_364, %padding_364, %ki0_364, %ki1_364, %ki2_364, %pad1_364, %out_v, %outidx_364) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 365: dilation=31 padding=124 kernel=(1,2,4) padding1=1
        %fidx_365 = arith.constant 365 : index
        %bidx_365 = arith.constant 616 : index
        %smidx_365 = arith.constant 1036 : index
        %ssidx_365 = arith.constant 1456 : index
        %bias_365 = memref.load %in_v[%bidx_365] : memref<1511xi32>
        %smean_365 = memref.load %in_v[%smidx_365] : memref<1511xi32>
        %sscale_365 = memref.load %in_v[%ssidx_365] : memref<1511xi32>
        %dilation_365 = arith.constant 31 : i32
        %padding_365 = arith.constant 124 : i32
        %ki0_365 = arith.constant 1 : i32
        %ki1_365 = arith.constant 2 : i32
        %ki2_365 = arith.constant 4 : i32
        %pad1_365 = arith.constant 1 : i32
        %outidx_365 = arith.constant 365 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_365, %smean_365, %sscale_365, %dilation_365, %padding_365, %ki0_365, %ki1_365, %ki2_365, %pad1_365, %out_v, %outidx_365) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 366: dilation=31 padding=124 kernel=(1,2,5) padding1=0
        %fidx_366 = arith.constant 366 : index
        %bidx_366 = arith.constant 617 : index
        %smidx_366 = arith.constant 1037 : index
        %ssidx_366 = arith.constant 1457 : index
        %bias_366 = memref.load %in_v[%bidx_366] : memref<1511xi32>
        %smean_366 = memref.load %in_v[%smidx_366] : memref<1511xi32>
        %sscale_366 = memref.load %in_v[%ssidx_366] : memref<1511xi32>
        %dilation_366 = arith.constant 31 : i32
        %padding_366 = arith.constant 124 : i32
        %ki0_366 = arith.constant 1 : i32
        %ki1_366 = arith.constant 2 : i32
        %ki2_366 = arith.constant 5 : i32
        %pad1_366 = arith.constant 0 : i32
        %outidx_366 = arith.constant 366 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_366, %smean_366, %sscale_366, %dilation_366, %padding_366, %ki0_366, %ki1_366, %ki2_366, %pad1_366, %out_v, %outidx_366) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 367: dilation=31 padding=124 kernel=(1,2,6) padding1=1
        %fidx_367 = arith.constant 367 : index
        %bidx_367 = arith.constant 618 : index
        %smidx_367 = arith.constant 1038 : index
        %ssidx_367 = arith.constant 1458 : index
        %bias_367 = memref.load %in_v[%bidx_367] : memref<1511xi32>
        %smean_367 = memref.load %in_v[%smidx_367] : memref<1511xi32>
        %sscale_367 = memref.load %in_v[%ssidx_367] : memref<1511xi32>
        %dilation_367 = arith.constant 31 : i32
        %padding_367 = arith.constant 124 : i32
        %ki0_367 = arith.constant 1 : i32
        %ki1_367 = arith.constant 2 : i32
        %ki2_367 = arith.constant 6 : i32
        %pad1_367 = arith.constant 1 : i32
        %outidx_367 = arith.constant 367 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_367, %smean_367, %sscale_367, %dilation_367, %padding_367, %ki0_367, %ki1_367, %ki2_367, %pad1_367, %out_v, %outidx_367) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 368: dilation=31 padding=124 kernel=(1,2,7) padding1=0
        %fidx_368 = arith.constant 368 : index
        %bidx_368 = arith.constant 619 : index
        %smidx_368 = arith.constant 1039 : index
        %ssidx_368 = arith.constant 1459 : index
        %bias_368 = memref.load %in_v[%bidx_368] : memref<1511xi32>
        %smean_368 = memref.load %in_v[%smidx_368] : memref<1511xi32>
        %sscale_368 = memref.load %in_v[%ssidx_368] : memref<1511xi32>
        %dilation_368 = arith.constant 31 : i32
        %padding_368 = arith.constant 124 : i32
        %ki0_368 = arith.constant 1 : i32
        %ki1_368 = arith.constant 2 : i32
        %ki2_368 = arith.constant 7 : i32
        %pad1_368 = arith.constant 0 : i32
        %outidx_368 = arith.constant 368 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_368, %smean_368, %sscale_368, %dilation_368, %padding_368, %ki0_368, %ki1_368, %ki2_368, %pad1_368, %out_v, %outidx_368) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 369: dilation=31 padding=124 kernel=(1,2,8) padding1=1
        %fidx_369 = arith.constant 369 : index
        %bidx_369 = arith.constant 620 : index
        %smidx_369 = arith.constant 1040 : index
        %ssidx_369 = arith.constant 1460 : index
        %bias_369 = memref.load %in_v[%bidx_369] : memref<1511xi32>
        %smean_369 = memref.load %in_v[%smidx_369] : memref<1511xi32>
        %sscale_369 = memref.load %in_v[%ssidx_369] : memref<1511xi32>
        %dilation_369 = arith.constant 31 : i32
        %padding_369 = arith.constant 124 : i32
        %ki0_369 = arith.constant 1 : i32
        %ki1_369 = arith.constant 2 : i32
        %ki2_369 = arith.constant 8 : i32
        %pad1_369 = arith.constant 1 : i32
        %outidx_369 = arith.constant 369 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_369, %smean_369, %sscale_369, %dilation_369, %padding_369, %ki0_369, %ki1_369, %ki2_369, %pad1_369, %out_v, %outidx_369) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 370: dilation=31 padding=124 kernel=(1,3,4) padding1=0
        %fidx_370 = arith.constant 370 : index
        %bidx_370 = arith.constant 621 : index
        %smidx_370 = arith.constant 1041 : index
        %ssidx_370 = arith.constant 1461 : index
        %bias_370 = memref.load %in_v[%bidx_370] : memref<1511xi32>
        %smean_370 = memref.load %in_v[%smidx_370] : memref<1511xi32>
        %sscale_370 = memref.load %in_v[%ssidx_370] : memref<1511xi32>
        %dilation_370 = arith.constant 31 : i32
        %padding_370 = arith.constant 124 : i32
        %ki0_370 = arith.constant 1 : i32
        %ki1_370 = arith.constant 3 : i32
        %ki2_370 = arith.constant 4 : i32
        %pad1_370 = arith.constant 0 : i32
        %outidx_370 = arith.constant 370 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_370, %smean_370, %sscale_370, %dilation_370, %padding_370, %ki0_370, %ki1_370, %ki2_370, %pad1_370, %out_v, %outidx_370) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 371: dilation=31 padding=124 kernel=(1,3,5) padding1=1
        %fidx_371 = arith.constant 371 : index
        %bidx_371 = arith.constant 622 : index
        %smidx_371 = arith.constant 1042 : index
        %ssidx_371 = arith.constant 1462 : index
        %bias_371 = memref.load %in_v[%bidx_371] : memref<1511xi32>
        %smean_371 = memref.load %in_v[%smidx_371] : memref<1511xi32>
        %sscale_371 = memref.load %in_v[%ssidx_371] : memref<1511xi32>
        %dilation_371 = arith.constant 31 : i32
        %padding_371 = arith.constant 124 : i32
        %ki0_371 = arith.constant 1 : i32
        %ki1_371 = arith.constant 3 : i32
        %ki2_371 = arith.constant 5 : i32
        %pad1_371 = arith.constant 1 : i32
        %outidx_371 = arith.constant 371 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_371, %smean_371, %sscale_371, %dilation_371, %padding_371, %ki0_371, %ki1_371, %ki2_371, %pad1_371, %out_v, %outidx_371) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 372: dilation=31 padding=124 kernel=(1,3,6) padding1=0
        %fidx_372 = arith.constant 372 : index
        %bidx_372 = arith.constant 623 : index
        %smidx_372 = arith.constant 1043 : index
        %ssidx_372 = arith.constant 1463 : index
        %bias_372 = memref.load %in_v[%bidx_372] : memref<1511xi32>
        %smean_372 = memref.load %in_v[%smidx_372] : memref<1511xi32>
        %sscale_372 = memref.load %in_v[%ssidx_372] : memref<1511xi32>
        %dilation_372 = arith.constant 31 : i32
        %padding_372 = arith.constant 124 : i32
        %ki0_372 = arith.constant 1 : i32
        %ki1_372 = arith.constant 3 : i32
        %ki2_372 = arith.constant 6 : i32
        %pad1_372 = arith.constant 0 : i32
        %outidx_372 = arith.constant 372 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_372, %smean_372, %sscale_372, %dilation_372, %padding_372, %ki0_372, %ki1_372, %ki2_372, %pad1_372, %out_v, %outidx_372) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 373: dilation=31 padding=124 kernel=(1,3,7) padding1=1
        %fidx_373 = arith.constant 373 : index
        %bidx_373 = arith.constant 624 : index
        %smidx_373 = arith.constant 1044 : index
        %ssidx_373 = arith.constant 1464 : index
        %bias_373 = memref.load %in_v[%bidx_373] : memref<1511xi32>
        %smean_373 = memref.load %in_v[%smidx_373] : memref<1511xi32>
        %sscale_373 = memref.load %in_v[%ssidx_373] : memref<1511xi32>
        %dilation_373 = arith.constant 31 : i32
        %padding_373 = arith.constant 124 : i32
        %ki0_373 = arith.constant 1 : i32
        %ki1_373 = arith.constant 3 : i32
        %ki2_373 = arith.constant 7 : i32
        %pad1_373 = arith.constant 1 : i32
        %outidx_373 = arith.constant 373 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_373, %smean_373, %sscale_373, %dilation_373, %padding_373, %ki0_373, %ki1_373, %ki2_373, %pad1_373, %out_v, %outidx_373) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 374: dilation=31 padding=124 kernel=(1,3,8) padding1=0
        %fidx_374 = arith.constant 374 : index
        %bidx_374 = arith.constant 625 : index
        %smidx_374 = arith.constant 1045 : index
        %ssidx_374 = arith.constant 1465 : index
        %bias_374 = memref.load %in_v[%bidx_374] : memref<1511xi32>
        %smean_374 = memref.load %in_v[%smidx_374] : memref<1511xi32>
        %sscale_374 = memref.load %in_v[%ssidx_374] : memref<1511xi32>
        %dilation_374 = arith.constant 31 : i32
        %padding_374 = arith.constant 124 : i32
        %ki0_374 = arith.constant 1 : i32
        %ki1_374 = arith.constant 3 : i32
        %ki2_374 = arith.constant 8 : i32
        %pad1_374 = arith.constant 0 : i32
        %outidx_374 = arith.constant 374 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_374, %smean_374, %sscale_374, %dilation_374, %padding_374, %ki0_374, %ki1_374, %ki2_374, %pad1_374, %out_v, %outidx_374) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 375: dilation=31 padding=124 kernel=(1,4,5) padding1=1
        %fidx_375 = arith.constant 375 : index
        %bidx_375 = arith.constant 626 : index
        %smidx_375 = arith.constant 1046 : index
        %ssidx_375 = arith.constant 1466 : index
        %bias_375 = memref.load %in_v[%bidx_375] : memref<1511xi32>
        %smean_375 = memref.load %in_v[%smidx_375] : memref<1511xi32>
        %sscale_375 = memref.load %in_v[%ssidx_375] : memref<1511xi32>
        %dilation_375 = arith.constant 31 : i32
        %padding_375 = arith.constant 124 : i32
        %ki0_375 = arith.constant 1 : i32
        %ki1_375 = arith.constant 4 : i32
        %ki2_375 = arith.constant 5 : i32
        %pad1_375 = arith.constant 1 : i32
        %outidx_375 = arith.constant 375 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_375, %smean_375, %sscale_375, %dilation_375, %padding_375, %ki0_375, %ki1_375, %ki2_375, %pad1_375, %out_v, %outidx_375) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 376: dilation=31 padding=124 kernel=(1,4,6) padding1=0
        %fidx_376 = arith.constant 376 : index
        %bidx_376 = arith.constant 627 : index
        %smidx_376 = arith.constant 1047 : index
        %ssidx_376 = arith.constant 1467 : index
        %bias_376 = memref.load %in_v[%bidx_376] : memref<1511xi32>
        %smean_376 = memref.load %in_v[%smidx_376] : memref<1511xi32>
        %sscale_376 = memref.load %in_v[%ssidx_376] : memref<1511xi32>
        %dilation_376 = arith.constant 31 : i32
        %padding_376 = arith.constant 124 : i32
        %ki0_376 = arith.constant 1 : i32
        %ki1_376 = arith.constant 4 : i32
        %ki2_376 = arith.constant 6 : i32
        %pad1_376 = arith.constant 0 : i32
        %outidx_376 = arith.constant 376 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_376, %smean_376, %sscale_376, %dilation_376, %padding_376, %ki0_376, %ki1_376, %ki2_376, %pad1_376, %out_v, %outidx_376) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 377: dilation=31 padding=124 kernel=(1,4,7) padding1=1
        %fidx_377 = arith.constant 377 : index
        %bidx_377 = arith.constant 628 : index
        %smidx_377 = arith.constant 1048 : index
        %ssidx_377 = arith.constant 1468 : index
        %bias_377 = memref.load %in_v[%bidx_377] : memref<1511xi32>
        %smean_377 = memref.load %in_v[%smidx_377] : memref<1511xi32>
        %sscale_377 = memref.load %in_v[%ssidx_377] : memref<1511xi32>
        %dilation_377 = arith.constant 31 : i32
        %padding_377 = arith.constant 124 : i32
        %ki0_377 = arith.constant 1 : i32
        %ki1_377 = arith.constant 4 : i32
        %ki2_377 = arith.constant 7 : i32
        %pad1_377 = arith.constant 1 : i32
        %outidx_377 = arith.constant 377 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_377, %smean_377, %sscale_377, %dilation_377, %padding_377, %ki0_377, %ki1_377, %ki2_377, %pad1_377, %out_v, %outidx_377) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 378: dilation=31 padding=124 kernel=(1,4,8) padding1=0
        %fidx_378 = arith.constant 378 : index
        %bidx_378 = arith.constant 629 : index
        %smidx_378 = arith.constant 1049 : index
        %ssidx_378 = arith.constant 1469 : index
        %bias_378 = memref.load %in_v[%bidx_378] : memref<1511xi32>
        %smean_378 = memref.load %in_v[%smidx_378] : memref<1511xi32>
        %sscale_378 = memref.load %in_v[%ssidx_378] : memref<1511xi32>
        %dilation_378 = arith.constant 31 : i32
        %padding_378 = arith.constant 124 : i32
        %ki0_378 = arith.constant 1 : i32
        %ki1_378 = arith.constant 4 : i32
        %ki2_378 = arith.constant 8 : i32
        %pad1_378 = arith.constant 0 : i32
        %outidx_378 = arith.constant 378 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_378, %smean_378, %sscale_378, %dilation_378, %padding_378, %ki0_378, %ki1_378, %ki2_378, %pad1_378, %out_v, %outidx_378) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 379: dilation=31 padding=124 kernel=(1,5,6) padding1=1
        %fidx_379 = arith.constant 379 : index
        %bidx_379 = arith.constant 630 : index
        %smidx_379 = arith.constant 1050 : index
        %ssidx_379 = arith.constant 1470 : index
        %bias_379 = memref.load %in_v[%bidx_379] : memref<1511xi32>
        %smean_379 = memref.load %in_v[%smidx_379] : memref<1511xi32>
        %sscale_379 = memref.load %in_v[%ssidx_379] : memref<1511xi32>
        %dilation_379 = arith.constant 31 : i32
        %padding_379 = arith.constant 124 : i32
        %ki0_379 = arith.constant 1 : i32
        %ki1_379 = arith.constant 5 : i32
        %ki2_379 = arith.constant 6 : i32
        %pad1_379 = arith.constant 1 : i32
        %outidx_379 = arith.constant 379 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_379, %smean_379, %sscale_379, %dilation_379, %padding_379, %ki0_379, %ki1_379, %ki2_379, %pad1_379, %out_v, %outidx_379) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 380: dilation=31 padding=124 kernel=(1,5,7) padding1=0
        %fidx_380 = arith.constant 380 : index
        %bidx_380 = arith.constant 631 : index
        %smidx_380 = arith.constant 1051 : index
        %ssidx_380 = arith.constant 1471 : index
        %bias_380 = memref.load %in_v[%bidx_380] : memref<1511xi32>
        %smean_380 = memref.load %in_v[%smidx_380] : memref<1511xi32>
        %sscale_380 = memref.load %in_v[%ssidx_380] : memref<1511xi32>
        %dilation_380 = arith.constant 31 : i32
        %padding_380 = arith.constant 124 : i32
        %ki0_380 = arith.constant 1 : i32
        %ki1_380 = arith.constant 5 : i32
        %ki2_380 = arith.constant 7 : i32
        %pad1_380 = arith.constant 0 : i32
        %outidx_380 = arith.constant 380 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_380, %smean_380, %sscale_380, %dilation_380, %padding_380, %ki0_380, %ki1_380, %ki2_380, %pad1_380, %out_v, %outidx_380) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 381: dilation=31 padding=124 kernel=(1,5,8) padding1=1
        %fidx_381 = arith.constant 381 : index
        %bidx_381 = arith.constant 632 : index
        %smidx_381 = arith.constant 1052 : index
        %ssidx_381 = arith.constant 1472 : index
        %bias_381 = memref.load %in_v[%bidx_381] : memref<1511xi32>
        %smean_381 = memref.load %in_v[%smidx_381] : memref<1511xi32>
        %sscale_381 = memref.load %in_v[%ssidx_381] : memref<1511xi32>
        %dilation_381 = arith.constant 31 : i32
        %padding_381 = arith.constant 124 : i32
        %ki0_381 = arith.constant 1 : i32
        %ki1_381 = arith.constant 5 : i32
        %ki2_381 = arith.constant 8 : i32
        %pad1_381 = arith.constant 1 : i32
        %outidx_381 = arith.constant 381 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_381, %smean_381, %sscale_381, %dilation_381, %padding_381, %ki0_381, %ki1_381, %ki2_381, %pad1_381, %out_v, %outidx_381) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 382: dilation=31 padding=124 kernel=(1,6,7) padding1=0
        %fidx_382 = arith.constant 382 : index
        %bidx_382 = arith.constant 633 : index
        %smidx_382 = arith.constant 1053 : index
        %ssidx_382 = arith.constant 1473 : index
        %bias_382 = memref.load %in_v[%bidx_382] : memref<1511xi32>
        %smean_382 = memref.load %in_v[%smidx_382] : memref<1511xi32>
        %sscale_382 = memref.load %in_v[%ssidx_382] : memref<1511xi32>
        %dilation_382 = arith.constant 31 : i32
        %padding_382 = arith.constant 124 : i32
        %ki0_382 = arith.constant 1 : i32
        %ki1_382 = arith.constant 6 : i32
        %ki2_382 = arith.constant 7 : i32
        %pad1_382 = arith.constant 0 : i32
        %outidx_382 = arith.constant 382 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_382, %smean_382, %sscale_382, %dilation_382, %padding_382, %ki0_382, %ki1_382, %ki2_382, %pad1_382, %out_v, %outidx_382) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 383: dilation=31 padding=124 kernel=(1,6,8) padding1=1
        %fidx_383 = arith.constant 383 : index
        %bidx_383 = arith.constant 634 : index
        %smidx_383 = arith.constant 1054 : index
        %ssidx_383 = arith.constant 1474 : index
        %bias_383 = memref.load %in_v[%bidx_383] : memref<1511xi32>
        %smean_383 = memref.load %in_v[%smidx_383] : memref<1511xi32>
        %sscale_383 = memref.load %in_v[%ssidx_383] : memref<1511xi32>
        %dilation_383 = arith.constant 31 : i32
        %padding_383 = arith.constant 124 : i32
        %ki0_383 = arith.constant 1 : i32
        %ki1_383 = arith.constant 6 : i32
        %ki2_383 = arith.constant 8 : i32
        %pad1_383 = arith.constant 1 : i32
        %outidx_383 = arith.constant 383 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_383, %smean_383, %sscale_383, %dilation_383, %padding_383, %ki0_383, %ki1_383, %ki2_383, %pad1_383, %out_v, %outidx_383) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 384: dilation=31 padding=124 kernel=(1,7,8) padding1=0
        %fidx_384 = arith.constant 384 : index
        %bidx_384 = arith.constant 635 : index
        %smidx_384 = arith.constant 1055 : index
        %ssidx_384 = arith.constant 1475 : index
        %bias_384 = memref.load %in_v[%bidx_384] : memref<1511xi32>
        %smean_384 = memref.load %in_v[%smidx_384] : memref<1511xi32>
        %sscale_384 = memref.load %in_v[%ssidx_384] : memref<1511xi32>
        %dilation_384 = arith.constant 31 : i32
        %padding_384 = arith.constant 124 : i32
        %ki0_384 = arith.constant 1 : i32
        %ki1_384 = arith.constant 7 : i32
        %ki2_384 = arith.constant 8 : i32
        %pad1_384 = arith.constant 0 : i32
        %outidx_384 = arith.constant 384 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_384, %smean_384, %sscale_384, %dilation_384, %padding_384, %ki0_384, %ki1_384, %ki2_384, %pad1_384, %out_v, %outidx_384) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 385: dilation=31 padding=124 kernel=(2,3,4) padding1=1
        %fidx_385 = arith.constant 385 : index
        %bidx_385 = arith.constant 636 : index
        %smidx_385 = arith.constant 1056 : index
        %ssidx_385 = arith.constant 1476 : index
        %bias_385 = memref.load %in_v[%bidx_385] : memref<1511xi32>
        %smean_385 = memref.load %in_v[%smidx_385] : memref<1511xi32>
        %sscale_385 = memref.load %in_v[%ssidx_385] : memref<1511xi32>
        %dilation_385 = arith.constant 31 : i32
        %padding_385 = arith.constant 124 : i32
        %ki0_385 = arith.constant 2 : i32
        %ki1_385 = arith.constant 3 : i32
        %ki2_385 = arith.constant 4 : i32
        %pad1_385 = arith.constant 1 : i32
        %outidx_385 = arith.constant 385 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_385, %smean_385, %sscale_385, %dilation_385, %padding_385, %ki0_385, %ki1_385, %ki2_385, %pad1_385, %out_v, %outidx_385) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 386: dilation=31 padding=124 kernel=(2,3,5) padding1=0
        %fidx_386 = arith.constant 386 : index
        %bidx_386 = arith.constant 637 : index
        %smidx_386 = arith.constant 1057 : index
        %ssidx_386 = arith.constant 1477 : index
        %bias_386 = memref.load %in_v[%bidx_386] : memref<1511xi32>
        %smean_386 = memref.load %in_v[%smidx_386] : memref<1511xi32>
        %sscale_386 = memref.load %in_v[%ssidx_386] : memref<1511xi32>
        %dilation_386 = arith.constant 31 : i32
        %padding_386 = arith.constant 124 : i32
        %ki0_386 = arith.constant 2 : i32
        %ki1_386 = arith.constant 3 : i32
        %ki2_386 = arith.constant 5 : i32
        %pad1_386 = arith.constant 0 : i32
        %outidx_386 = arith.constant 386 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_386, %smean_386, %sscale_386, %dilation_386, %padding_386, %ki0_386, %ki1_386, %ki2_386, %pad1_386, %out_v, %outidx_386) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 387: dilation=31 padding=124 kernel=(2,3,6) padding1=1
        %fidx_387 = arith.constant 387 : index
        %bidx_387 = arith.constant 638 : index
        %smidx_387 = arith.constant 1058 : index
        %ssidx_387 = arith.constant 1478 : index
        %bias_387 = memref.load %in_v[%bidx_387] : memref<1511xi32>
        %smean_387 = memref.load %in_v[%smidx_387] : memref<1511xi32>
        %sscale_387 = memref.load %in_v[%ssidx_387] : memref<1511xi32>
        %dilation_387 = arith.constant 31 : i32
        %padding_387 = arith.constant 124 : i32
        %ki0_387 = arith.constant 2 : i32
        %ki1_387 = arith.constant 3 : i32
        %ki2_387 = arith.constant 6 : i32
        %pad1_387 = arith.constant 1 : i32
        %outidx_387 = arith.constant 387 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_387, %smean_387, %sscale_387, %dilation_387, %padding_387, %ki0_387, %ki1_387, %ki2_387, %pad1_387, %out_v, %outidx_387) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 388: dilation=31 padding=124 kernel=(2,3,7) padding1=0
        %fidx_388 = arith.constant 388 : index
        %bidx_388 = arith.constant 639 : index
        %smidx_388 = arith.constant 1059 : index
        %ssidx_388 = arith.constant 1479 : index
        %bias_388 = memref.load %in_v[%bidx_388] : memref<1511xi32>
        %smean_388 = memref.load %in_v[%smidx_388] : memref<1511xi32>
        %sscale_388 = memref.load %in_v[%ssidx_388] : memref<1511xi32>
        %dilation_388 = arith.constant 31 : i32
        %padding_388 = arith.constant 124 : i32
        %ki0_388 = arith.constant 2 : i32
        %ki1_388 = arith.constant 3 : i32
        %ki2_388 = arith.constant 7 : i32
        %pad1_388 = arith.constant 0 : i32
        %outidx_388 = arith.constant 388 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_388, %smean_388, %sscale_388, %dilation_388, %padding_388, %ki0_388, %ki1_388, %ki2_388, %pad1_388, %out_v, %outidx_388) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 389: dilation=31 padding=124 kernel=(2,3,8) padding1=1
        %fidx_389 = arith.constant 389 : index
        %bidx_389 = arith.constant 640 : index
        %smidx_389 = arith.constant 1060 : index
        %ssidx_389 = arith.constant 1480 : index
        %bias_389 = memref.load %in_v[%bidx_389] : memref<1511xi32>
        %smean_389 = memref.load %in_v[%smidx_389] : memref<1511xi32>
        %sscale_389 = memref.load %in_v[%ssidx_389] : memref<1511xi32>
        %dilation_389 = arith.constant 31 : i32
        %padding_389 = arith.constant 124 : i32
        %ki0_389 = arith.constant 2 : i32
        %ki1_389 = arith.constant 3 : i32
        %ki2_389 = arith.constant 8 : i32
        %pad1_389 = arith.constant 1 : i32
        %outidx_389 = arith.constant 389 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_389, %smean_389, %sscale_389, %dilation_389, %padding_389, %ki0_389, %ki1_389, %ki2_389, %pad1_389, %out_v, %outidx_389) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 390: dilation=31 padding=124 kernel=(2,4,5) padding1=0
        %fidx_390 = arith.constant 390 : index
        %bidx_390 = arith.constant 641 : index
        %smidx_390 = arith.constant 1061 : index
        %ssidx_390 = arith.constant 1481 : index
        %bias_390 = memref.load %in_v[%bidx_390] : memref<1511xi32>
        %smean_390 = memref.load %in_v[%smidx_390] : memref<1511xi32>
        %sscale_390 = memref.load %in_v[%ssidx_390] : memref<1511xi32>
        %dilation_390 = arith.constant 31 : i32
        %padding_390 = arith.constant 124 : i32
        %ki0_390 = arith.constant 2 : i32
        %ki1_390 = arith.constant 4 : i32
        %ki2_390 = arith.constant 5 : i32
        %pad1_390 = arith.constant 0 : i32
        %outidx_390 = arith.constant 390 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_390, %smean_390, %sscale_390, %dilation_390, %padding_390, %ki0_390, %ki1_390, %ki2_390, %pad1_390, %out_v, %outidx_390) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 391: dilation=31 padding=124 kernel=(2,4,6) padding1=1
        %fidx_391 = arith.constant 391 : index
        %bidx_391 = arith.constant 642 : index
        %smidx_391 = arith.constant 1062 : index
        %ssidx_391 = arith.constant 1482 : index
        %bias_391 = memref.load %in_v[%bidx_391] : memref<1511xi32>
        %smean_391 = memref.load %in_v[%smidx_391] : memref<1511xi32>
        %sscale_391 = memref.load %in_v[%ssidx_391] : memref<1511xi32>
        %dilation_391 = arith.constant 31 : i32
        %padding_391 = arith.constant 124 : i32
        %ki0_391 = arith.constant 2 : i32
        %ki1_391 = arith.constant 4 : i32
        %ki2_391 = arith.constant 6 : i32
        %pad1_391 = arith.constant 1 : i32
        %outidx_391 = arith.constant 391 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_391, %smean_391, %sscale_391, %dilation_391, %padding_391, %ki0_391, %ki1_391, %ki2_391, %pad1_391, %out_v, %outidx_391) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 392: dilation=31 padding=124 kernel=(2,4,7) padding1=0
        %fidx_392 = arith.constant 392 : index
        %bidx_392 = arith.constant 643 : index
        %smidx_392 = arith.constant 1063 : index
        %ssidx_392 = arith.constant 1483 : index
        %bias_392 = memref.load %in_v[%bidx_392] : memref<1511xi32>
        %smean_392 = memref.load %in_v[%smidx_392] : memref<1511xi32>
        %sscale_392 = memref.load %in_v[%ssidx_392] : memref<1511xi32>
        %dilation_392 = arith.constant 31 : i32
        %padding_392 = arith.constant 124 : i32
        %ki0_392 = arith.constant 2 : i32
        %ki1_392 = arith.constant 4 : i32
        %ki2_392 = arith.constant 7 : i32
        %pad1_392 = arith.constant 0 : i32
        %outidx_392 = arith.constant 392 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_392, %smean_392, %sscale_392, %dilation_392, %padding_392, %ki0_392, %ki1_392, %ki2_392, %pad1_392, %out_v, %outidx_392) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 393: dilation=31 padding=124 kernel=(2,4,8) padding1=1
        %fidx_393 = arith.constant 393 : index
        %bidx_393 = arith.constant 644 : index
        %smidx_393 = arith.constant 1064 : index
        %ssidx_393 = arith.constant 1484 : index
        %bias_393 = memref.load %in_v[%bidx_393] : memref<1511xi32>
        %smean_393 = memref.load %in_v[%smidx_393] : memref<1511xi32>
        %sscale_393 = memref.load %in_v[%ssidx_393] : memref<1511xi32>
        %dilation_393 = arith.constant 31 : i32
        %padding_393 = arith.constant 124 : i32
        %ki0_393 = arith.constant 2 : i32
        %ki1_393 = arith.constant 4 : i32
        %ki2_393 = arith.constant 8 : i32
        %pad1_393 = arith.constant 1 : i32
        %outidx_393 = arith.constant 393 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_393, %smean_393, %sscale_393, %dilation_393, %padding_393, %ki0_393, %ki1_393, %ki2_393, %pad1_393, %out_v, %outidx_393) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 394: dilation=31 padding=124 kernel=(2,5,6) padding1=0
        %fidx_394 = arith.constant 394 : index
        %bidx_394 = arith.constant 645 : index
        %smidx_394 = arith.constant 1065 : index
        %ssidx_394 = arith.constant 1485 : index
        %bias_394 = memref.load %in_v[%bidx_394] : memref<1511xi32>
        %smean_394 = memref.load %in_v[%smidx_394] : memref<1511xi32>
        %sscale_394 = memref.load %in_v[%ssidx_394] : memref<1511xi32>
        %dilation_394 = arith.constant 31 : i32
        %padding_394 = arith.constant 124 : i32
        %ki0_394 = arith.constant 2 : i32
        %ki1_394 = arith.constant 5 : i32
        %ki2_394 = arith.constant 6 : i32
        %pad1_394 = arith.constant 0 : i32
        %outidx_394 = arith.constant 394 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_394, %smean_394, %sscale_394, %dilation_394, %padding_394, %ki0_394, %ki1_394, %ki2_394, %pad1_394, %out_v, %outidx_394) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 395: dilation=31 padding=124 kernel=(2,5,7) padding1=1
        %fidx_395 = arith.constant 395 : index
        %bidx_395 = arith.constant 646 : index
        %smidx_395 = arith.constant 1066 : index
        %ssidx_395 = arith.constant 1486 : index
        %bias_395 = memref.load %in_v[%bidx_395] : memref<1511xi32>
        %smean_395 = memref.load %in_v[%smidx_395] : memref<1511xi32>
        %sscale_395 = memref.load %in_v[%ssidx_395] : memref<1511xi32>
        %dilation_395 = arith.constant 31 : i32
        %padding_395 = arith.constant 124 : i32
        %ki0_395 = arith.constant 2 : i32
        %ki1_395 = arith.constant 5 : i32
        %ki2_395 = arith.constant 7 : i32
        %pad1_395 = arith.constant 1 : i32
        %outidx_395 = arith.constant 395 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_395, %smean_395, %sscale_395, %dilation_395, %padding_395, %ki0_395, %ki1_395, %ki2_395, %pad1_395, %out_v, %outidx_395) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 396: dilation=31 padding=124 kernel=(2,5,8) padding1=0
        %fidx_396 = arith.constant 396 : index
        %bidx_396 = arith.constant 647 : index
        %smidx_396 = arith.constant 1067 : index
        %ssidx_396 = arith.constant 1487 : index
        %bias_396 = memref.load %in_v[%bidx_396] : memref<1511xi32>
        %smean_396 = memref.load %in_v[%smidx_396] : memref<1511xi32>
        %sscale_396 = memref.load %in_v[%ssidx_396] : memref<1511xi32>
        %dilation_396 = arith.constant 31 : i32
        %padding_396 = arith.constant 124 : i32
        %ki0_396 = arith.constant 2 : i32
        %ki1_396 = arith.constant 5 : i32
        %ki2_396 = arith.constant 8 : i32
        %pad1_396 = arith.constant 0 : i32
        %outidx_396 = arith.constant 396 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_396, %smean_396, %sscale_396, %dilation_396, %padding_396, %ki0_396, %ki1_396, %ki2_396, %pad1_396, %out_v, %outidx_396) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 397: dilation=31 padding=124 kernel=(2,6,7) padding1=1
        %fidx_397 = arith.constant 397 : index
        %bidx_397 = arith.constant 648 : index
        %smidx_397 = arith.constant 1068 : index
        %ssidx_397 = arith.constant 1488 : index
        %bias_397 = memref.load %in_v[%bidx_397] : memref<1511xi32>
        %smean_397 = memref.load %in_v[%smidx_397] : memref<1511xi32>
        %sscale_397 = memref.load %in_v[%ssidx_397] : memref<1511xi32>
        %dilation_397 = arith.constant 31 : i32
        %padding_397 = arith.constant 124 : i32
        %ki0_397 = arith.constant 2 : i32
        %ki1_397 = arith.constant 6 : i32
        %ki2_397 = arith.constant 7 : i32
        %pad1_397 = arith.constant 1 : i32
        %outidx_397 = arith.constant 397 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_397, %smean_397, %sscale_397, %dilation_397, %padding_397, %ki0_397, %ki1_397, %ki2_397, %pad1_397, %out_v, %outidx_397) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 398: dilation=31 padding=124 kernel=(2,6,8) padding1=0
        %fidx_398 = arith.constant 398 : index
        %bidx_398 = arith.constant 649 : index
        %smidx_398 = arith.constant 1069 : index
        %ssidx_398 = arith.constant 1489 : index
        %bias_398 = memref.load %in_v[%bidx_398] : memref<1511xi32>
        %smean_398 = memref.load %in_v[%smidx_398] : memref<1511xi32>
        %sscale_398 = memref.load %in_v[%ssidx_398] : memref<1511xi32>
        %dilation_398 = arith.constant 31 : i32
        %padding_398 = arith.constant 124 : i32
        %ki0_398 = arith.constant 2 : i32
        %ki1_398 = arith.constant 6 : i32
        %ki2_398 = arith.constant 8 : i32
        %pad1_398 = arith.constant 0 : i32
        %outidx_398 = arith.constant 398 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_398, %smean_398, %sscale_398, %dilation_398, %padding_398, %ki0_398, %ki1_398, %ki2_398, %pad1_398, %out_v, %outidx_398) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 399: dilation=31 padding=124 kernel=(2,7,8) padding1=1
        %fidx_399 = arith.constant 399 : index
        %bidx_399 = arith.constant 650 : index
        %smidx_399 = arith.constant 1070 : index
        %ssidx_399 = arith.constant 1490 : index
        %bias_399 = memref.load %in_v[%bidx_399] : memref<1511xi32>
        %smean_399 = memref.load %in_v[%smidx_399] : memref<1511xi32>
        %sscale_399 = memref.load %in_v[%ssidx_399] : memref<1511xi32>
        %dilation_399 = arith.constant 31 : i32
        %padding_399 = arith.constant 124 : i32
        %ki0_399 = arith.constant 2 : i32
        %ki1_399 = arith.constant 7 : i32
        %ki2_399 = arith.constant 8 : i32
        %pad1_399 = arith.constant 1 : i32
        %outidx_399 = arith.constant 399 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_399, %smean_399, %sscale_399, %dilation_399, %padding_399, %ki0_399, %ki1_399, %ki2_399, %pad1_399, %out_v, %outidx_399) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 400: dilation=31 padding=124 kernel=(3,4,5) padding1=0
        %fidx_400 = arith.constant 400 : index
        %bidx_400 = arith.constant 651 : index
        %smidx_400 = arith.constant 1071 : index
        %ssidx_400 = arith.constant 1491 : index
        %bias_400 = memref.load %in_v[%bidx_400] : memref<1511xi32>
        %smean_400 = memref.load %in_v[%smidx_400] : memref<1511xi32>
        %sscale_400 = memref.load %in_v[%ssidx_400] : memref<1511xi32>
        %dilation_400 = arith.constant 31 : i32
        %padding_400 = arith.constant 124 : i32
        %ki0_400 = arith.constant 3 : i32
        %ki1_400 = arith.constant 4 : i32
        %ki2_400 = arith.constant 5 : i32
        %pad1_400 = arith.constant 0 : i32
        %outidx_400 = arith.constant 400 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_400, %smean_400, %sscale_400, %dilation_400, %padding_400, %ki0_400, %ki1_400, %ki2_400, %pad1_400, %out_v, %outidx_400) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 401: dilation=31 padding=124 kernel=(3,4,6) padding1=1
        %fidx_401 = arith.constant 401 : index
        %bidx_401 = arith.constant 652 : index
        %smidx_401 = arith.constant 1072 : index
        %ssidx_401 = arith.constant 1492 : index
        %bias_401 = memref.load %in_v[%bidx_401] : memref<1511xi32>
        %smean_401 = memref.load %in_v[%smidx_401] : memref<1511xi32>
        %sscale_401 = memref.load %in_v[%ssidx_401] : memref<1511xi32>
        %dilation_401 = arith.constant 31 : i32
        %padding_401 = arith.constant 124 : i32
        %ki0_401 = arith.constant 3 : i32
        %ki1_401 = arith.constant 4 : i32
        %ki2_401 = arith.constant 6 : i32
        %pad1_401 = arith.constant 1 : i32
        %outidx_401 = arith.constant 401 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_401, %smean_401, %sscale_401, %dilation_401, %padding_401, %ki0_401, %ki1_401, %ki2_401, %pad1_401, %out_v, %outidx_401) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 402: dilation=31 padding=124 kernel=(3,4,7) padding1=0
        %fidx_402 = arith.constant 402 : index
        %bidx_402 = arith.constant 653 : index
        %smidx_402 = arith.constant 1073 : index
        %ssidx_402 = arith.constant 1493 : index
        %bias_402 = memref.load %in_v[%bidx_402] : memref<1511xi32>
        %smean_402 = memref.load %in_v[%smidx_402] : memref<1511xi32>
        %sscale_402 = memref.load %in_v[%ssidx_402] : memref<1511xi32>
        %dilation_402 = arith.constant 31 : i32
        %padding_402 = arith.constant 124 : i32
        %ki0_402 = arith.constant 3 : i32
        %ki1_402 = arith.constant 4 : i32
        %ki2_402 = arith.constant 7 : i32
        %pad1_402 = arith.constant 0 : i32
        %outidx_402 = arith.constant 402 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_402, %smean_402, %sscale_402, %dilation_402, %padding_402, %ki0_402, %ki1_402, %ki2_402, %pad1_402, %out_v, %outidx_402) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 403: dilation=31 padding=124 kernel=(3,4,8) padding1=1
        %fidx_403 = arith.constant 403 : index
        %bidx_403 = arith.constant 654 : index
        %smidx_403 = arith.constant 1074 : index
        %ssidx_403 = arith.constant 1494 : index
        %bias_403 = memref.load %in_v[%bidx_403] : memref<1511xi32>
        %smean_403 = memref.load %in_v[%smidx_403] : memref<1511xi32>
        %sscale_403 = memref.load %in_v[%ssidx_403] : memref<1511xi32>
        %dilation_403 = arith.constant 31 : i32
        %padding_403 = arith.constant 124 : i32
        %ki0_403 = arith.constant 3 : i32
        %ki1_403 = arith.constant 4 : i32
        %ki2_403 = arith.constant 8 : i32
        %pad1_403 = arith.constant 1 : i32
        %outidx_403 = arith.constant 403 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_403, %smean_403, %sscale_403, %dilation_403, %padding_403, %ki0_403, %ki1_403, %ki2_403, %pad1_403, %out_v, %outidx_403) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 404: dilation=31 padding=124 kernel=(3,5,6) padding1=0
        %fidx_404 = arith.constant 404 : index
        %bidx_404 = arith.constant 655 : index
        %smidx_404 = arith.constant 1075 : index
        %ssidx_404 = arith.constant 1495 : index
        %bias_404 = memref.load %in_v[%bidx_404] : memref<1511xi32>
        %smean_404 = memref.load %in_v[%smidx_404] : memref<1511xi32>
        %sscale_404 = memref.load %in_v[%ssidx_404] : memref<1511xi32>
        %dilation_404 = arith.constant 31 : i32
        %padding_404 = arith.constant 124 : i32
        %ki0_404 = arith.constant 3 : i32
        %ki1_404 = arith.constant 5 : i32
        %ki2_404 = arith.constant 6 : i32
        %pad1_404 = arith.constant 0 : i32
        %outidx_404 = arith.constant 404 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_404, %smean_404, %sscale_404, %dilation_404, %padding_404, %ki0_404, %ki1_404, %ki2_404, %pad1_404, %out_v, %outidx_404) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 405: dilation=31 padding=124 kernel=(3,5,7) padding1=1
        %fidx_405 = arith.constant 405 : index
        %bidx_405 = arith.constant 656 : index
        %smidx_405 = arith.constant 1076 : index
        %ssidx_405 = arith.constant 1496 : index
        %bias_405 = memref.load %in_v[%bidx_405] : memref<1511xi32>
        %smean_405 = memref.load %in_v[%smidx_405] : memref<1511xi32>
        %sscale_405 = memref.load %in_v[%ssidx_405] : memref<1511xi32>
        %dilation_405 = arith.constant 31 : i32
        %padding_405 = arith.constant 124 : i32
        %ki0_405 = arith.constant 3 : i32
        %ki1_405 = arith.constant 5 : i32
        %ki2_405 = arith.constant 7 : i32
        %pad1_405 = arith.constant 1 : i32
        %outidx_405 = arith.constant 405 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_405, %smean_405, %sscale_405, %dilation_405, %padding_405, %ki0_405, %ki1_405, %ki2_405, %pad1_405, %out_v, %outidx_405) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 406: dilation=31 padding=124 kernel=(3,5,8) padding1=0
        %fidx_406 = arith.constant 406 : index
        %bidx_406 = arith.constant 657 : index
        %smidx_406 = arith.constant 1077 : index
        %ssidx_406 = arith.constant 1497 : index
        %bias_406 = memref.load %in_v[%bidx_406] : memref<1511xi32>
        %smean_406 = memref.load %in_v[%smidx_406] : memref<1511xi32>
        %sscale_406 = memref.load %in_v[%ssidx_406] : memref<1511xi32>
        %dilation_406 = arith.constant 31 : i32
        %padding_406 = arith.constant 124 : i32
        %ki0_406 = arith.constant 3 : i32
        %ki1_406 = arith.constant 5 : i32
        %ki2_406 = arith.constant 8 : i32
        %pad1_406 = arith.constant 0 : i32
        %outidx_406 = arith.constant 406 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_406, %smean_406, %sscale_406, %dilation_406, %padding_406, %ki0_406, %ki1_406, %ki2_406, %pad1_406, %out_v, %outidx_406) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 407: dilation=31 padding=124 kernel=(3,6,7) padding1=1
        %fidx_407 = arith.constant 407 : index
        %bidx_407 = arith.constant 658 : index
        %smidx_407 = arith.constant 1078 : index
        %ssidx_407 = arith.constant 1498 : index
        %bias_407 = memref.load %in_v[%bidx_407] : memref<1511xi32>
        %smean_407 = memref.load %in_v[%smidx_407] : memref<1511xi32>
        %sscale_407 = memref.load %in_v[%ssidx_407] : memref<1511xi32>
        %dilation_407 = arith.constant 31 : i32
        %padding_407 = arith.constant 124 : i32
        %ki0_407 = arith.constant 3 : i32
        %ki1_407 = arith.constant 6 : i32
        %ki2_407 = arith.constant 7 : i32
        %pad1_407 = arith.constant 1 : i32
        %outidx_407 = arith.constant 407 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_407, %smean_407, %sscale_407, %dilation_407, %padding_407, %ki0_407, %ki1_407, %ki2_407, %pad1_407, %out_v, %outidx_407) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 408: dilation=31 padding=124 kernel=(3,6,8) padding1=0
        %fidx_408 = arith.constant 408 : index
        %bidx_408 = arith.constant 659 : index
        %smidx_408 = arith.constant 1079 : index
        %ssidx_408 = arith.constant 1499 : index
        %bias_408 = memref.load %in_v[%bidx_408] : memref<1511xi32>
        %smean_408 = memref.load %in_v[%smidx_408] : memref<1511xi32>
        %sscale_408 = memref.load %in_v[%ssidx_408] : memref<1511xi32>
        %dilation_408 = arith.constant 31 : i32
        %padding_408 = arith.constant 124 : i32
        %ki0_408 = arith.constant 3 : i32
        %ki1_408 = arith.constant 6 : i32
        %ki2_408 = arith.constant 8 : i32
        %pad1_408 = arith.constant 0 : i32
        %outidx_408 = arith.constant 408 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_408, %smean_408, %sscale_408, %dilation_408, %padding_408, %ki0_408, %ki1_408, %ki2_408, %pad1_408, %out_v, %outidx_408) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 409: dilation=31 padding=124 kernel=(3,7,8) padding1=1
        %fidx_409 = arith.constant 409 : index
        %bidx_409 = arith.constant 660 : index
        %smidx_409 = arith.constant 1080 : index
        %ssidx_409 = arith.constant 1500 : index
        %bias_409 = memref.load %in_v[%bidx_409] : memref<1511xi32>
        %smean_409 = memref.load %in_v[%smidx_409] : memref<1511xi32>
        %sscale_409 = memref.load %in_v[%ssidx_409] : memref<1511xi32>
        %dilation_409 = arith.constant 31 : i32
        %padding_409 = arith.constant 124 : i32
        %ki0_409 = arith.constant 3 : i32
        %ki1_409 = arith.constant 7 : i32
        %ki2_409 = arith.constant 8 : i32
        %pad1_409 = arith.constant 1 : i32
        %outidx_409 = arith.constant 409 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_409, %smean_409, %sscale_409, %dilation_409, %padding_409, %ki0_409, %ki1_409, %ki2_409, %pad1_409, %out_v, %outidx_409) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 410: dilation=31 padding=124 kernel=(4,5,6) padding1=0
        %fidx_410 = arith.constant 410 : index
        %bidx_410 = arith.constant 661 : index
        %smidx_410 = arith.constant 1081 : index
        %ssidx_410 = arith.constant 1501 : index
        %bias_410 = memref.load %in_v[%bidx_410] : memref<1511xi32>
        %smean_410 = memref.load %in_v[%smidx_410] : memref<1511xi32>
        %sscale_410 = memref.load %in_v[%ssidx_410] : memref<1511xi32>
        %dilation_410 = arith.constant 31 : i32
        %padding_410 = arith.constant 124 : i32
        %ki0_410 = arith.constant 4 : i32
        %ki1_410 = arith.constant 5 : i32
        %ki2_410 = arith.constant 6 : i32
        %pad1_410 = arith.constant 0 : i32
        %outidx_410 = arith.constant 410 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_410, %smean_410, %sscale_410, %dilation_410, %padding_410, %ki0_410, %ki1_410, %ki2_410, %pad1_410, %out_v, %outidx_410) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 411: dilation=31 padding=124 kernel=(4,5,7) padding1=1
        %fidx_411 = arith.constant 411 : index
        %bidx_411 = arith.constant 662 : index
        %smidx_411 = arith.constant 1082 : index
        %ssidx_411 = arith.constant 1502 : index
        %bias_411 = memref.load %in_v[%bidx_411] : memref<1511xi32>
        %smean_411 = memref.load %in_v[%smidx_411] : memref<1511xi32>
        %sscale_411 = memref.load %in_v[%ssidx_411] : memref<1511xi32>
        %dilation_411 = arith.constant 31 : i32
        %padding_411 = arith.constant 124 : i32
        %ki0_411 = arith.constant 4 : i32
        %ki1_411 = arith.constant 5 : i32
        %ki2_411 = arith.constant 7 : i32
        %pad1_411 = arith.constant 1 : i32
        %outidx_411 = arith.constant 411 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_411, %smean_411, %sscale_411, %dilation_411, %padding_411, %ki0_411, %ki1_411, %ki2_411, %pad1_411, %out_v, %outidx_411) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 412: dilation=31 padding=124 kernel=(4,5,8) padding1=0
        %fidx_412 = arith.constant 412 : index
        %bidx_412 = arith.constant 663 : index
        %smidx_412 = arith.constant 1083 : index
        %ssidx_412 = arith.constant 1503 : index
        %bias_412 = memref.load %in_v[%bidx_412] : memref<1511xi32>
        %smean_412 = memref.load %in_v[%smidx_412] : memref<1511xi32>
        %sscale_412 = memref.load %in_v[%ssidx_412] : memref<1511xi32>
        %dilation_412 = arith.constant 31 : i32
        %padding_412 = arith.constant 124 : i32
        %ki0_412 = arith.constant 4 : i32
        %ki1_412 = arith.constant 5 : i32
        %ki2_412 = arith.constant 8 : i32
        %pad1_412 = arith.constant 0 : i32
        %outidx_412 = arith.constant 412 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_412, %smean_412, %sscale_412, %dilation_412, %padding_412, %ki0_412, %ki1_412, %ki2_412, %pad1_412, %out_v, %outidx_412) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 413: dilation=31 padding=124 kernel=(4,6,7) padding1=1
        %fidx_413 = arith.constant 413 : index
        %bidx_413 = arith.constant 664 : index
        %smidx_413 = arith.constant 1084 : index
        %ssidx_413 = arith.constant 1504 : index
        %bias_413 = memref.load %in_v[%bidx_413] : memref<1511xi32>
        %smean_413 = memref.load %in_v[%smidx_413] : memref<1511xi32>
        %sscale_413 = memref.load %in_v[%ssidx_413] : memref<1511xi32>
        %dilation_413 = arith.constant 31 : i32
        %padding_413 = arith.constant 124 : i32
        %ki0_413 = arith.constant 4 : i32
        %ki1_413 = arith.constant 6 : i32
        %ki2_413 = arith.constant 7 : i32
        %pad1_413 = arith.constant 1 : i32
        %outidx_413 = arith.constant 413 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_413, %smean_413, %sscale_413, %dilation_413, %padding_413, %ki0_413, %ki1_413, %ki2_413, %pad1_413, %out_v, %outidx_413) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 414: dilation=31 padding=124 kernel=(4,6,8) padding1=0
        %fidx_414 = arith.constant 414 : index
        %bidx_414 = arith.constant 665 : index
        %smidx_414 = arith.constant 1085 : index
        %ssidx_414 = arith.constant 1505 : index
        %bias_414 = memref.load %in_v[%bidx_414] : memref<1511xi32>
        %smean_414 = memref.load %in_v[%smidx_414] : memref<1511xi32>
        %sscale_414 = memref.load %in_v[%ssidx_414] : memref<1511xi32>
        %dilation_414 = arith.constant 31 : i32
        %padding_414 = arith.constant 124 : i32
        %ki0_414 = arith.constant 4 : i32
        %ki1_414 = arith.constant 6 : i32
        %ki2_414 = arith.constant 8 : i32
        %pad1_414 = arith.constant 0 : i32
        %outidx_414 = arith.constant 414 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_414, %smean_414, %sscale_414, %dilation_414, %padding_414, %ki0_414, %ki1_414, %ki2_414, %pad1_414, %out_v, %outidx_414) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 415: dilation=31 padding=124 kernel=(4,7,8) padding1=1
        %fidx_415 = arith.constant 415 : index
        %bidx_415 = arith.constant 666 : index
        %smidx_415 = arith.constant 1086 : index
        %ssidx_415 = arith.constant 1506 : index
        %bias_415 = memref.load %in_v[%bidx_415] : memref<1511xi32>
        %smean_415 = memref.load %in_v[%smidx_415] : memref<1511xi32>
        %sscale_415 = memref.load %in_v[%ssidx_415] : memref<1511xi32>
        %dilation_415 = arith.constant 31 : i32
        %padding_415 = arith.constant 124 : i32
        %ki0_415 = arith.constant 4 : i32
        %ki1_415 = arith.constant 7 : i32
        %ki2_415 = arith.constant 8 : i32
        %pad1_415 = arith.constant 1 : i32
        %outidx_415 = arith.constant 415 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_415, %smean_415, %sscale_415, %dilation_415, %padding_415, %ki0_415, %ki1_415, %ki2_415, %pad1_415, %out_v, %outidx_415) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 416: dilation=31 padding=124 kernel=(5,6,7) padding1=0
        %fidx_416 = arith.constant 416 : index
        %bidx_416 = arith.constant 667 : index
        %smidx_416 = arith.constant 1087 : index
        %ssidx_416 = arith.constant 1507 : index
        %bias_416 = memref.load %in_v[%bidx_416] : memref<1511xi32>
        %smean_416 = memref.load %in_v[%smidx_416] : memref<1511xi32>
        %sscale_416 = memref.load %in_v[%ssidx_416] : memref<1511xi32>
        %dilation_416 = arith.constant 31 : i32
        %padding_416 = arith.constant 124 : i32
        %ki0_416 = arith.constant 5 : i32
        %ki1_416 = arith.constant 6 : i32
        %ki2_416 = arith.constant 7 : i32
        %pad1_416 = arith.constant 0 : i32
        %outidx_416 = arith.constant 416 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_416, %smean_416, %sscale_416, %dilation_416, %padding_416, %ki0_416, %ki1_416, %ki2_416, %pad1_416, %out_v, %outidx_416) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 417: dilation=31 padding=124 kernel=(5,6,8) padding1=1
        %fidx_417 = arith.constant 417 : index
        %bidx_417 = arith.constant 668 : index
        %smidx_417 = arith.constant 1088 : index
        %ssidx_417 = arith.constant 1508 : index
        %bias_417 = memref.load %in_v[%bidx_417] : memref<1511xi32>
        %smean_417 = memref.load %in_v[%smidx_417] : memref<1511xi32>
        %sscale_417 = memref.load %in_v[%ssidx_417] : memref<1511xi32>
        %dilation_417 = arith.constant 31 : i32
        %padding_417 = arith.constant 124 : i32
        %ki0_417 = arith.constant 5 : i32
        %ki1_417 = arith.constant 6 : i32
        %ki2_417 = arith.constant 8 : i32
        %pad1_417 = arith.constant 1 : i32
        %outidx_417 = arith.constant 417 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_417, %smean_417, %sscale_417, %dilation_417, %padding_417, %ki0_417, %ki1_417, %ki2_417, %pad1_417, %out_v, %outidx_417) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 418: dilation=31 padding=124 kernel=(5,7,8) padding1=0
        %fidx_418 = arith.constant 418 : index
        %bidx_418 = arith.constant 669 : index
        %smidx_418 = arith.constant 1089 : index
        %ssidx_418 = arith.constant 1509 : index
        %bias_418 = memref.load %in_v[%bidx_418] : memref<1511xi32>
        %smean_418 = memref.load %in_v[%smidx_418] : memref<1511xi32>
        %sscale_418 = memref.load %in_v[%ssidx_418] : memref<1511xi32>
        %dilation_418 = arith.constant 31 : i32
        %padding_418 = arith.constant 124 : i32
        %ki0_418 = arith.constant 5 : i32
        %ki1_418 = arith.constant 7 : i32
        %ki2_418 = arith.constant 8 : i32
        %pad1_418 = arith.constant 0 : i32
        %outidx_418 = arith.constant 418 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_418, %smean_418, %sscale_418, %dilation_418, %padding_418, %ki0_418, %ki1_418, %ki2_418, %pad1_418, %out_v, %outidx_418) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        // feature 419: dilation=31 padding=124 kernel=(6,7,8) padding1=1
        %fidx_419 = arith.constant 419 : index
        %bidx_419 = arith.constant 670 : index
        %smidx_419 = arith.constant 1090 : index
        %ssidx_419 = arith.constant 1510 : index
        %bias_419 = memref.load %in_v[%bidx_419] : memref<1511xi32>
        %smean_419 = memref.load %in_v[%smidx_419] : memref<1511xi32>
        %sscale_419 = memref.load %in_v[%ssidx_419] : memref<1511xi32>
        %dilation_419 = arith.constant 31 : i32
        %padding_419 = arith.constant 124 : i32
        %ki0_419 = arith.constant 6 : i32
        %ki1_419 = arith.constant 7 : i32
        %ki2_419 = arith.constant 8 : i32
        %pad1_419 = arith.constant 1 : i32
        %outidx_419 = arith.constant 419 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_419, %smean_419, %sscale_419, %dilation_419, %padding_419, %ki0_419, %ki1_419, %ki2_419, %pad1_419, %out_v, %outidx_419) : (memref<1511xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<420xi32>, i32) -> ()

        aie.objectfifo.release @in_mem_1(Consume, 1)
        aie.objectfifo.release @out_mem_1(Produce, 1)
      }
      aie.end
    } {stack_size = 4096 : i32, link_with = "minirocket_transform_kernel.o"}

    // --- HOST RUNTIME SEQUENCE ---
    aie.runtime_sequence(%arg0: memref<3022xi32>, %arg1: memref<840xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 1511][0, 0, 0, 1]) {id = 0 : i64, metadata = @in_0} : memref<3022xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 420][0, 0, 0, 1]) {id = 1 : i64, metadata = @out_0} : memref<840xi32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 1511][1, 1, 1, 1511][0, 0, 0, 1]) {id = 2 : i64, metadata = @in_1} : memref<3022xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 420][1, 1, 1, 420][0, 0, 0, 1]) {id = 3 : i64, metadata = @out_1} : memref<840xi32>
      aiex.npu.dma_wait {symbol = @out_0}
      aiex.npu.dma_wait {symbol = @out_1}
    }
  }
}