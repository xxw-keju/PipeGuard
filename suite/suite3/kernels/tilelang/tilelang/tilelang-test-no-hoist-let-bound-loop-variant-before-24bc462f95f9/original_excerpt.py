    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        for i in range(128):
            idx = i % 2
            if idx == 0:
                B[i] = A[i]
