    @T.prim_func
    def vector_add(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        with T.Kernel(128, threads=128) as i:
            B[i] = A[i] + 1.0
