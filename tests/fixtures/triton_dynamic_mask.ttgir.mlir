#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @dynamic_mask(%base: !tt.ptr<i8>, %mask: tensor<16xi1, #blocked>) {
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %base_vector = tt.splat %base : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>, #blocked>
    %pointer = tt.addptr %base_vector, %range : tensor<16x!tt.ptr<i8>, #blocked>, tensor<16xi32, #blocked>
    %loaded = tt.load %pointer, %mask : tensor<16x!tt.ptr<i8>, #blocked>
    tt.return
  }
}
