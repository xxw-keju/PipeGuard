#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_store_roundtrip(%descriptor: !tt.tensordesc<64x64xf32, #shared>) {
    %zero = arith.constant 0 : i32
    %payload = arith.constant dense<0.000000e+00> : tensor<64x64xf32, #blocked>
    %source = ttg.local_alloc %payload {allocation.offset = 0 : i32} : (tensor<64x64xf32, #blocked>) -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_copy_local_to_global %descriptor[%zero, %zero] %source : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.async_tma_store_wait {pendings = 0 : i32}
    tt.return
  }
}
