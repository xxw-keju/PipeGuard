    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((1,), T.float32),
    ):
        for i in range(128):
            pos = C[0]
            if pos >= T.float32(0):
                B[i] = A[i]
