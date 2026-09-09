    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            if i < 64:
                B[i] = A[i]
