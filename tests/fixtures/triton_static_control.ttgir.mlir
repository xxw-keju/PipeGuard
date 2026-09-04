#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @static_control() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c3 = arith.constant 3 : index
    %pure = arith.addi %c1, %c1 : index
    %dst = ttg.local_alloc : () -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    scf.for %iteration = %c0 to %c3 step %c1 {
      %loaded = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
      scf.yield
    }
    %true = arith.constant true
    %false = arith.constant false
    scf.if %true {
      %then_loaded = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    } else {
      %unreachable0 = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
      %unreachable1 = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    }
    scf.if %false {
      %unreachable2 = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
      %unreachable3 = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    } else {
      %else_loaded = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    }
    tt.return
  }
}
