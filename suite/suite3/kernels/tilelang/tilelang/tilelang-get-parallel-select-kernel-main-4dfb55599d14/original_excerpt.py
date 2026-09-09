    @T.prim_func
    def main(
        A: T.Tensor[(1024,), T.float32],
        B: T.Tensor[(1024,), T.float32],
    ):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(1024):
                B[i] = T.Select(i < 512, A[i], 0.0)
