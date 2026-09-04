#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @initialized_local() {
    %payload = arith.constant dense<0> : tensor<16xi8, #blocked>
    %dst = ttg.local_alloc %payload : (tensor<16xi8, #blocked>) -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %loaded = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    tt.return
  }
}
