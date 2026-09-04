#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @masked_async(%base: !tt.ptr<i8>) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %eight = arith.constant 8 : i32
    %eights = tt.splat %eight : i32 -> tensor<16xi32, #blocked>
    %mask = arith.cmpi slt, %range, %eights : tensor<16xi32, #blocked>
    %base_vector = tt.splat %base : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>, #blocked>
    %src = tt.addptr %base_vector, %range : tensor<16x!tt.ptr<i8>, #blocked>, tensor<16xi32, #blocked>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %src, %dst mask %mask : tensor<16x!tt.ptr<i8>, #blocked> -> <16xi8, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %group {num = 0 : i32}
    %loaded = ttg.local_load %dst token %wait : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    tt.return
  }
}
