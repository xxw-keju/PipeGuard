    @T.prim_func
    def main(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(1, threads=128):
            S = T.alloc_shared((128,), T.float32)
            for i in T.Parallel(128):
                S[i] = A[i]
            B[0] = S[0]
