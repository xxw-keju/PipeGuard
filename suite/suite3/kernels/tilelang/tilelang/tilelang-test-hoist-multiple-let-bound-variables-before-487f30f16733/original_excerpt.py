    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((2,), T.float32),
    ):
        for i in range(128):
            x = C[0]
            y = C[1]
            if x + y >= T.float32(0):
                B[i] = A[i]
