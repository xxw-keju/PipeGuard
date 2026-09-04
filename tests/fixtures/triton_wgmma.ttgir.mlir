#blockedA = #ttg.blocked<{sizePerThread = [16, 1], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
#blockedB = #ttg.blocked<{sizePerThread = [1, 16], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [0, 1]}>
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
#sharedA = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 16}>
#sharedB = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @wgmma_roundtrip() {
    %a_tensor = arith.constant dense<0.000000e+00> : tensor<128x64xf16, #blockedA>
    %b_tensor = arith.constant dense<0.000000e+00> : tensor<64x64xf16, #blockedB>
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %a = ttg.local_alloc %a_tensor : (tensor<128x64xf16, #blockedA>) -> !ttg.memdesc<128x64xf16, #sharedA, #smem>
    %b = ttg.local_alloc %b_tensor : (tensor<64x64xf16, #blockedB>) -> !ttg.memdesc<64x64xf16, #sharedB, #smem>
    %dot = ttng.warp_group_dot %a, %b, %acc {isAsync = true} : !ttg.memdesc<128x64xf16, #sharedA, #smem> * !ttg.memdesc<64x64xf16, #sharedB, #smem> -> tensor<128x64xf32, #mma>
    %waited = ttng.warp_group_dot_wait %dot {pendings = 0 : i32} : tensor<128x64xf32, #mma>
    tt.return
  }
}
