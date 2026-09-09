    @T.prim_func
    def main(out: T.Tensor((128,), T.float32)):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((128,), T.float32)
            for i in T.Parallel(128):
                smem[i] = T.float32(1.0)
            T.fill(smem, T.cast(0, T.float32))
            T.copy(smem, out)
