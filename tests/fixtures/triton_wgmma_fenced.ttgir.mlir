#blocked = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#shared1 = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wgmma_roundtrip() {
    %cst = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blocked>
    %cst_0 = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blocked1>
    %cst_1 = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %0 = ttg.local_alloc %cst : (tensor<128x64xf16, #blocked>) -> !ttg.memdesc<128x64xf16, #shared, #smem>
    %1 = ttg.local_alloc %cst_0 : (tensor<64x64xf16, #blocked1>) -> !ttg.memdesc<64x64xf16, #shared1, #smem>
    ttng.fence_async_shared {bCluster = false}
    %2 = ttng.warp_group_dot %0, %1, %cst_1 {isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
    %3 = ttng.warp_group_dot_wait %2 {pendings = 0 : i32} : tensor<128x64xf32, #mma>
    tt.return
  }
}

