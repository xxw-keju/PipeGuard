    @T.prim_func
    def main(out: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            local = T.alloc_local((32,), T.float32)
            for i in T.serial(32):
                local[i] = T.float32(i)
            for i in T.Parallel(32):
                out[i] = local[i]
