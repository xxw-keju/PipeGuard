    @T.prim_func
    def main(
        A: T.Tensor[(256,), T.float32],
        B: T.Tensor[(256,), T.float32],
    ):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(256):
                B[i] = T.Select(A[i] > T.float32(0), A[i], T.float32(0))
