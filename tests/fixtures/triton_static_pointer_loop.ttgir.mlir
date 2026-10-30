#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @static_pointer_loop(%base: !tt.ptr<i8>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %base_vector = tt.splat %base : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>, #blocked>
    scf.for %iteration = %c0 to %c3 step %c1 {
      %iteration_i32 = arith.index_cast %iteration : index to i32
      %offset = tt.splat %iteration_i32 : i32 -> tensor<16xi32, #blocked>
      %pointer = tt.addptr %base_vector, %offset : tensor<16x!tt.ptr<i8>, #blocked>, tensor<16xi32, #blocked>
      %loaded = tt.load %pointer : tensor<16x!tt.ptr<i8>, #blocked>
      scf.yield
    }
    tt.return
  }
}
