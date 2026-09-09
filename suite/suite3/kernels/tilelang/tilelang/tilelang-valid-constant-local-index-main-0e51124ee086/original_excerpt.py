    @T.prim_func
    def main(out: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            scale_local = T.alloc_local((1,), T.float32)
            scale_local[0] = T.float32(2)
            for i in T.Parallel(32):
                out[i] = scale_local[0]
