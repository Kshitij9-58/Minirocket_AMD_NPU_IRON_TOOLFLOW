module {
  aie.device(npu1) {
    func.func private @zero_i32_scalar(memref<1xi32>)
    func.func private @minirocket_feature_transform(memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32)

    %shim_noc_tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %shim_noc_tile_1_0 = aie.tile(1, 0)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)

    // --- ROUTING: COLUMN 0 (transform tile 0) ---
    aie.objectfifo @in_0(%shim_noc_tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<275xi32>>
    aie.objectfifo @in_mem_0(%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<275xi32>>
    aie.objectfifo.link [@in_0] -> [@in_mem_0]([] [])
    aie.objectfifo @out_mem_0(%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out_0(%mem_tile_0_1, {%shim_noc_tile_0_0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@out_mem_0] -> [@out_0]([] [])

    // --- ROUTING: COLUMN 1 (transform tile 1) ---
    aie.objectfifo @in_1(%shim_noc_tile_1_0, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<275xi32>>
    aie.objectfifo @in_mem_1(%mem_tile_1_1, {%tile_1_2}, 2 : i32) : !aie.objectfifo<memref<275xi32>>
    aie.objectfifo.link [@in_1] -> [@in_mem_1]([] [])
    aie.objectfifo @out_mem_1(%tile_1_2, {%mem_tile_1_1}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @out_1(%mem_tile_1_1, {%shim_noc_tile_1_0}, 2 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo.link [@out_mem_1] -> [@out_1]([] [])

    // --- CORES ---
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c_series_off = arith.constant 0 : i32

      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %in_elem = aie.objectfifo.acquire @in_mem_0(Consume, 1) : !aie.objectfifosubview<memref<275xi32>>
        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<275xi32>> -> memref<275xi32>
        %out = aie.objectfifo.acquire @out_mem_0(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>

        // feature 0: dilation=1 padding=4 kernel=(0,1,2) padding1=0
        %fidx_0 = arith.constant 0 : index
        %bidx_0 = arith.constant 251 : index
        %smidx_0 = arith.constant 259 : index
        %ssidx_0 = arith.constant 267 : index
        %bias_0 = memref.load %in_v[%bidx_0] : memref<275xi32>
        %smean_0 = memref.load %in_v[%smidx_0] : memref<275xi32>
        %sscale_0 = memref.load %in_v[%ssidx_0] : memref<275xi32>
        %dilation_0 = arith.constant 1 : i32
        %padding_0 = arith.constant 4 : i32
        %ki0_0 = arith.constant 0 : i32
        %ki1_0 = arith.constant 1 : i32
        %ki2_0 = arith.constant 2 : i32
        %pad1_0 = arith.constant 0 : i32
        %outidx_0 = arith.constant 0 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_0, %smean_0, %sscale_0, %dilation_0, %padding_0, %ki0_0, %ki1_0, %ki2_0, %pad1_0, %out_v, %outidx_0) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 1: dilation=1 padding=4 kernel=(0,1,2) padding1=0
        %fidx_1 = arith.constant 1 : index
        %bidx_1 = arith.constant 252 : index
        %smidx_1 = arith.constant 260 : index
        %ssidx_1 = arith.constant 268 : index
        %bias_1 = memref.load %in_v[%bidx_1] : memref<275xi32>
        %smean_1 = memref.load %in_v[%smidx_1] : memref<275xi32>
        %sscale_1 = memref.load %in_v[%ssidx_1] : memref<275xi32>
        %dilation_1 = arith.constant 1 : i32
        %padding_1 = arith.constant 4 : i32
        %ki0_1 = arith.constant 0 : i32
        %ki1_1 = arith.constant 1 : i32
        %ki2_1 = arith.constant 2 : i32
        %pad1_1 = arith.constant 0 : i32
        %outidx_1 = arith.constant 1 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_1, %smean_1, %sscale_1, %dilation_1, %padding_1, %ki0_1, %ki1_1, %ki2_1, %pad1_1, %out_v, %outidx_1) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 2: dilation=1 padding=4 kernel=(0,1,3) padding1=1
        %fidx_2 = arith.constant 2 : index
        %bidx_2 = arith.constant 253 : index
        %smidx_2 = arith.constant 261 : index
        %ssidx_2 = arith.constant 269 : index
        %bias_2 = memref.load %in_v[%bidx_2] : memref<275xi32>
        %smean_2 = memref.load %in_v[%smidx_2] : memref<275xi32>
        %sscale_2 = memref.load %in_v[%ssidx_2] : memref<275xi32>
        %dilation_2 = arith.constant 1 : i32
        %padding_2 = arith.constant 4 : i32
        %ki0_2 = arith.constant 0 : i32
        %ki1_2 = arith.constant 1 : i32
        %ki2_2 = arith.constant 3 : i32
        %pad1_2 = arith.constant 1 : i32
        %outidx_2 = arith.constant 2 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_2, %smean_2, %sscale_2, %dilation_2, %padding_2, %ki0_2, %ki1_2, %ki2_2, %pad1_2, %out_v, %outidx_2) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 3: dilation=1 padding=4 kernel=(0,1,3) padding1=1
        %fidx_3 = arith.constant 3 : index
        %bidx_3 = arith.constant 254 : index
        %smidx_3 = arith.constant 262 : index
        %ssidx_3 = arith.constant 270 : index
        %bias_3 = memref.load %in_v[%bidx_3] : memref<275xi32>
        %smean_3 = memref.load %in_v[%smidx_3] : memref<275xi32>
        %sscale_3 = memref.load %in_v[%ssidx_3] : memref<275xi32>
        %dilation_3 = arith.constant 1 : i32
        %padding_3 = arith.constant 4 : i32
        %ki0_3 = arith.constant 0 : i32
        %ki1_3 = arith.constant 1 : i32
        %ki2_3 = arith.constant 3 : i32
        %pad1_3 = arith.constant 1 : i32
        %outidx_3 = arith.constant 3 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_3, %smean_3, %sscale_3, %dilation_3, %padding_3, %ki0_3, %ki1_3, %ki2_3, %pad1_3, %out_v, %outidx_3) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 4: dilation=1 padding=4 kernel=(0,1,4) padding1=0
        %fidx_4 = arith.constant 4 : index
        %bidx_4 = arith.constant 255 : index
        %smidx_4 = arith.constant 263 : index
        %ssidx_4 = arith.constant 271 : index
        %bias_4 = memref.load %in_v[%bidx_4] : memref<275xi32>
        %smean_4 = memref.load %in_v[%smidx_4] : memref<275xi32>
        %sscale_4 = memref.load %in_v[%ssidx_4] : memref<275xi32>
        %dilation_4 = arith.constant 1 : i32
        %padding_4 = arith.constant 4 : i32
        %ki0_4 = arith.constant 0 : i32
        %ki1_4 = arith.constant 1 : i32
        %ki2_4 = arith.constant 4 : i32
        %pad1_4 = arith.constant 0 : i32
        %outidx_4 = arith.constant 4 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_4, %smean_4, %sscale_4, %dilation_4, %padding_4, %ki0_4, %ki1_4, %ki2_4, %pad1_4, %out_v, %outidx_4) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 5: dilation=1 padding=4 kernel=(0,1,4) padding1=0
        %fidx_5 = arith.constant 5 : index
        %bidx_5 = arith.constant 256 : index
        %smidx_5 = arith.constant 264 : index
        %ssidx_5 = arith.constant 272 : index
        %bias_5 = memref.load %in_v[%bidx_5] : memref<275xi32>
        %smean_5 = memref.load %in_v[%smidx_5] : memref<275xi32>
        %sscale_5 = memref.load %in_v[%ssidx_5] : memref<275xi32>
        %dilation_5 = arith.constant 1 : i32
        %padding_5 = arith.constant 4 : i32
        %ki0_5 = arith.constant 0 : i32
        %ki1_5 = arith.constant 1 : i32
        %ki2_5 = arith.constant 4 : i32
        %pad1_5 = arith.constant 0 : i32
        %outidx_5 = arith.constant 5 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_5, %smean_5, %sscale_5, %dilation_5, %padding_5, %ki0_5, %ki1_5, %ki2_5, %pad1_5, %out_v, %outidx_5) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 6: dilation=1 padding=4 kernel=(0,1,5) padding1=1
        %fidx_6 = arith.constant 6 : index
        %bidx_6 = arith.constant 257 : index
        %smidx_6 = arith.constant 265 : index
        %ssidx_6 = arith.constant 273 : index
        %bias_6 = memref.load %in_v[%bidx_6] : memref<275xi32>
        %smean_6 = memref.load %in_v[%smidx_6] : memref<275xi32>
        %sscale_6 = memref.load %in_v[%ssidx_6] : memref<275xi32>
        %dilation_6 = arith.constant 1 : i32
        %padding_6 = arith.constant 4 : i32
        %ki0_6 = arith.constant 0 : i32
        %ki1_6 = arith.constant 1 : i32
        %ki2_6 = arith.constant 5 : i32
        %pad1_6 = arith.constant 1 : i32
        %outidx_6 = arith.constant 6 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_6, %smean_6, %sscale_6, %dilation_6, %padding_6, %ki0_6, %ki1_6, %ki2_6, %pad1_6, %out_v, %outidx_6) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 7: dilation=1 padding=4 kernel=(0,1,5) padding1=1
        %fidx_7 = arith.constant 7 : index
        %bidx_7 = arith.constant 258 : index
        %smidx_7 = arith.constant 266 : index
        %ssidx_7 = arith.constant 274 : index
        %bias_7 = memref.load %in_v[%bidx_7] : memref<275xi32>
        %smean_7 = memref.load %in_v[%smidx_7] : memref<275xi32>
        %sscale_7 = memref.load %in_v[%ssidx_7] : memref<275xi32>
        %dilation_7 = arith.constant 1 : i32
        %padding_7 = arith.constant 4 : i32
        %ki0_7 = arith.constant 0 : i32
        %ki1_7 = arith.constant 1 : i32
        %ki2_7 = arith.constant 5 : i32
        %pad1_7 = arith.constant 1 : i32
        %outidx_7 = arith.constant 7 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_7, %smean_7, %sscale_7, %dilation_7, %padding_7, %ki0_7, %ki1_7, %ki2_7, %pad1_7, %out_v, %outidx_7) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        aie.objectfifo.release @in_mem_0(Consume, 1)
        aie.objectfifo.release @out_mem_0(Produce, 1)
      }
      aie.end
    } {stack_size = 4096 : i32}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c4294967295 = arith.constant 4294967295 : index
      %c1 = arith.constant 1 : index
      %c_series_off = arith.constant 0 : i32

      scf.for %arg0 = %c0 to %c4294967295 step %c1 {
        %in_elem = aie.objectfifo.acquire @in_mem_1(Consume, 1) : !aie.objectfifosubview<memref<275xi32>>
        %in_v = aie.objectfifo.subview.access %in_elem[0] : !aie.objectfifosubview<memref<275xi32>> -> memref<275xi32>
        %out = aie.objectfifo.acquire @out_mem_1(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %out_v = aie.objectfifo.subview.access %out[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>

        // feature 0: dilation=6 padding=24 kernel=(0,1,2) padding1=0
        %fidx_0 = arith.constant 0 : index
        %bidx_0 = arith.constant 251 : index
        %smidx_0 = arith.constant 259 : index
        %ssidx_0 = arith.constant 267 : index
        %bias_0 = memref.load %in_v[%bidx_0] : memref<275xi32>
        %smean_0 = memref.load %in_v[%smidx_0] : memref<275xi32>
        %sscale_0 = memref.load %in_v[%ssidx_0] : memref<275xi32>
        %dilation_0 = arith.constant 6 : i32
        %padding_0 = arith.constant 24 : i32
        %ki0_0 = arith.constant 0 : i32
        %ki1_0 = arith.constant 1 : i32
        %ki2_0 = arith.constant 2 : i32
        %pad1_0 = arith.constant 0 : i32
        %outidx_0 = arith.constant 0 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_0, %smean_0, %sscale_0, %dilation_0, %padding_0, %ki0_0, %ki1_0, %ki2_0, %pad1_0, %out_v, %outidx_0) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 1: dilation=6 padding=24 kernel=(0,1,3) padding1=1
        %fidx_1 = arith.constant 1 : index
        %bidx_1 = arith.constant 252 : index
        %smidx_1 = arith.constant 260 : index
        %ssidx_1 = arith.constant 268 : index
        %bias_1 = memref.load %in_v[%bidx_1] : memref<275xi32>
        %smean_1 = memref.load %in_v[%smidx_1] : memref<275xi32>
        %sscale_1 = memref.load %in_v[%ssidx_1] : memref<275xi32>
        %dilation_1 = arith.constant 6 : i32
        %padding_1 = arith.constant 24 : i32
        %ki0_1 = arith.constant 0 : i32
        %ki1_1 = arith.constant 1 : i32
        %ki2_1 = arith.constant 3 : i32
        %pad1_1 = arith.constant 1 : i32
        %outidx_1 = arith.constant 1 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_1, %smean_1, %sscale_1, %dilation_1, %padding_1, %ki0_1, %ki1_1, %ki2_1, %pad1_1, %out_v, %outidx_1) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 2: dilation=6 padding=24 kernel=(0,1,4) padding1=0
        %fidx_2 = arith.constant 2 : index
        %bidx_2 = arith.constant 253 : index
        %smidx_2 = arith.constant 261 : index
        %ssidx_2 = arith.constant 269 : index
        %bias_2 = memref.load %in_v[%bidx_2] : memref<275xi32>
        %smean_2 = memref.load %in_v[%smidx_2] : memref<275xi32>
        %sscale_2 = memref.load %in_v[%ssidx_2] : memref<275xi32>
        %dilation_2 = arith.constant 6 : i32
        %padding_2 = arith.constant 24 : i32
        %ki0_2 = arith.constant 0 : i32
        %ki1_2 = arith.constant 1 : i32
        %ki2_2 = arith.constant 4 : i32
        %pad1_2 = arith.constant 0 : i32
        %outidx_2 = arith.constant 2 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_2, %smean_2, %sscale_2, %dilation_2, %padding_2, %ki0_2, %ki1_2, %ki2_2, %pad1_2, %out_v, %outidx_2) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 3: dilation=6 padding=24 kernel=(0,1,5) padding1=1
        %fidx_3 = arith.constant 3 : index
        %bidx_3 = arith.constant 254 : index
        %smidx_3 = arith.constant 262 : index
        %ssidx_3 = arith.constant 270 : index
        %bias_3 = memref.load %in_v[%bidx_3] : memref<275xi32>
        %smean_3 = memref.load %in_v[%smidx_3] : memref<275xi32>
        %sscale_3 = memref.load %in_v[%ssidx_3] : memref<275xi32>
        %dilation_3 = arith.constant 6 : i32
        %padding_3 = arith.constant 24 : i32
        %ki0_3 = arith.constant 0 : i32
        %ki1_3 = arith.constant 1 : i32
        %ki2_3 = arith.constant 5 : i32
        %pad1_3 = arith.constant 1 : i32
        %outidx_3 = arith.constant 3 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_3, %smean_3, %sscale_3, %dilation_3, %padding_3, %ki0_3, %ki1_3, %ki2_3, %pad1_3, %out_v, %outidx_3) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 4: dilation=6 padding=24 kernel=(0,1,6) padding1=0
        %fidx_4 = arith.constant 4 : index
        %bidx_4 = arith.constant 255 : index
        %smidx_4 = arith.constant 263 : index
        %ssidx_4 = arith.constant 271 : index
        %bias_4 = memref.load %in_v[%bidx_4] : memref<275xi32>
        %smean_4 = memref.load %in_v[%smidx_4] : memref<275xi32>
        %sscale_4 = memref.load %in_v[%ssidx_4] : memref<275xi32>
        %dilation_4 = arith.constant 6 : i32
        %padding_4 = arith.constant 24 : i32
        %ki0_4 = arith.constant 0 : i32
        %ki1_4 = arith.constant 1 : i32
        %ki2_4 = arith.constant 6 : i32
        %pad1_4 = arith.constant 0 : i32
        %outidx_4 = arith.constant 4 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_4, %smean_4, %sscale_4, %dilation_4, %padding_4, %ki0_4, %ki1_4, %ki2_4, %pad1_4, %out_v, %outidx_4) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 5: dilation=6 padding=24 kernel=(0,1,7) padding1=1
        %fidx_5 = arith.constant 5 : index
        %bidx_5 = arith.constant 256 : index
        %smidx_5 = arith.constant 264 : index
        %ssidx_5 = arith.constant 272 : index
        %bias_5 = memref.load %in_v[%bidx_5] : memref<275xi32>
        %smean_5 = memref.load %in_v[%smidx_5] : memref<275xi32>
        %sscale_5 = memref.load %in_v[%ssidx_5] : memref<275xi32>
        %dilation_5 = arith.constant 6 : i32
        %padding_5 = arith.constant 24 : i32
        %ki0_5 = arith.constant 0 : i32
        %ki1_5 = arith.constant 1 : i32
        %ki2_5 = arith.constant 7 : i32
        %pad1_5 = arith.constant 1 : i32
        %outidx_5 = arith.constant 5 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_5, %smean_5, %sscale_5, %dilation_5, %padding_5, %ki0_5, %ki1_5, %ki2_5, %pad1_5, %out_v, %outidx_5) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 6: dilation=6 padding=24 kernel=(0,1,8) padding1=0
        %fidx_6 = arith.constant 6 : index
        %bidx_6 = arith.constant 257 : index
        %smidx_6 = arith.constant 265 : index
        %ssidx_6 = arith.constant 273 : index
        %bias_6 = memref.load %in_v[%bidx_6] : memref<275xi32>
        %smean_6 = memref.load %in_v[%smidx_6] : memref<275xi32>
        %sscale_6 = memref.load %in_v[%ssidx_6] : memref<275xi32>
        %dilation_6 = arith.constant 6 : i32
        %padding_6 = arith.constant 24 : i32
        %ki0_6 = arith.constant 0 : i32
        %ki1_6 = arith.constant 1 : i32
        %ki2_6 = arith.constant 8 : i32
        %pad1_6 = arith.constant 0 : i32
        %outidx_6 = arith.constant 6 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_6, %smean_6, %sscale_6, %dilation_6, %padding_6, %ki0_6, %ki1_6, %ki2_6, %pad1_6, %out_v, %outidx_6) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        // feature 7: dilation=6 padding=24 kernel=(0,2,3) padding1=1
        %fidx_7 = arith.constant 7 : index
        %bidx_7 = arith.constant 258 : index
        %smidx_7 = arith.constant 266 : index
        %ssidx_7 = arith.constant 274 : index
        %bias_7 = memref.load %in_v[%bidx_7] : memref<275xi32>
        %smean_7 = memref.load %in_v[%smidx_7] : memref<275xi32>
        %sscale_7 = memref.load %in_v[%ssidx_7] : memref<275xi32>
        %dilation_7 = arith.constant 6 : i32
        %padding_7 = arith.constant 24 : i32
        %ki0_7 = arith.constant 0 : i32
        %ki1_7 = arith.constant 2 : i32
        %ki2_7 = arith.constant 3 : i32
        %pad1_7 = arith.constant 1 : i32
        %outidx_7 = arith.constant 7 : i32
        func.call @minirocket_feature_transform(%in_v, %c_series_off, %bias_7, %smean_7, %sscale_7, %dilation_7, %padding_7, %ki0_7, %ki1_7, %ki2_7, %pad1_7, %out_v, %outidx_7) : (memref<275xi32>, i32, i32, i32, i32, i32, i32, i32, i32, i32, i32, memref<8xi32>, i32) -> ()

        aie.objectfifo.release @in_mem_1(Consume, 1)
        aie.objectfifo.release @out_mem_1(Produce, 1)
      }
      aie.end
    } {stack_size = 4096 : i32}

    // --- HOST RUNTIME SEQUENCE ---
    aie.runtime_sequence(%arg0: memref<550xi32>, %arg1: memref<16xi32>) {
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 0][1, 1, 1, 275][0, 0, 0, 1]) {id = 0 : i64, metadata = @in_0} : memref<550xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1]) {id = 1 : i64, metadata = @out_0} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%arg0[0, 0, 0, 275][1, 1, 1, 275][0, 0, 0, 1]) {id = 2 : i64, metadata = @in_1} : memref<550xi32>
      aiex.npu.dma_memcpy_nd(%arg1[0, 0, 0, 8][1, 1, 1, 8][0, 0, 0, 1]) {id = 3 : i64, metadata = @out_1} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @out_0}
      aiex.npu.dma_wait {symbol = @out_1}
    }
  }
}