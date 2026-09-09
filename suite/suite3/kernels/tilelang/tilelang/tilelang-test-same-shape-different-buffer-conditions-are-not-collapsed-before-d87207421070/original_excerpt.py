    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        P: T.Tensor((1,), T.int32),
        Q: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            if P[0] < 4:
                B[i] = A[i]
            if Q[0] < 4:
                C[i] = A[i] * T.float32(2.0)
