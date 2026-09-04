#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [1, 32], warpsPerCTA = [1, 4], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = false, elementBitWidth = 32}>
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @tma_load_roundtrip(%descriptor: !tt.tensordesc<64x64xf32, #shared>, %barrier_arg: !ttg.memdesc<1xi64, #barrier, #smem, mutable>) {
    %zero = arith.constant 0 : i32
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %destination = ttg.local_alloc : () -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.init_barrier %barrier_arg, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.barrier_expect %barrier_arg, 16384, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.async_tma_copy_global_to_local %descriptor[%zero, %zero] %destination, %barrier_arg, %true : !tt.tensordesc<64x64xf32, #shared>, !ttg.memdesc<1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>
    ttng.wait_barrier %barrier_arg, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %loaded = ttg.local_load %destination : !ttg.memdesc<64x64xf32, #shared, #smem, mutable> -> tensor<64x64xf32, #blocked>
    tt.return
  }
}
