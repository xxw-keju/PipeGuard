#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @warp_specialize_static() {
    %payload = arith.constant dense<0> : tensor<16xi8, #blocked>
    %buffer = ttg.local_alloc %payload : (tensor<16xi8, #blocked>) -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    ttg.warp_specialize(%buffer)
    default {
      %default_value = ttg.local_load %buffer : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
      ttg.warp_yield
    }
    partition0(%partition_buffer: !ttg.memdesc<16xi8, #shared, #smem, mutable>) num_warps(4) {
      %partition_value = ttg.local_load %partition_buffer : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
      ttg.warp_return
    } : (!ttg.memdesc<16xi8, #shared, #smem, mutable>) -> ()
    %after = ttg.local_load %buffer : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    tt.return
  }
}
