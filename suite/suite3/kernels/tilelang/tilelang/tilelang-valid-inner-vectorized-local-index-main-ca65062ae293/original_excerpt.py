    @T.prim_func
    def main(out: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            scratch = T.alloc_local((4,), T.float32)
            for i in T.Parallel(32):
                for lane in T.vectorized(4):
                    scratch[lane] = T.float32(lane)
                out[i] = scratch[0]
