#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_store_read_only(%arg0: !tt.tensordesc<64x64xf32, #shared>) {
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %0 = ttg.local_alloc %cst {allocation.offset = 0 : i32} : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.fence_async_shared {bCluster = false}
    ttng.async_tma_copy_local_to_global %arg0[%c0_i32, %c0_i32] %0 : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32, read_only}
    tt.return
  }
}

