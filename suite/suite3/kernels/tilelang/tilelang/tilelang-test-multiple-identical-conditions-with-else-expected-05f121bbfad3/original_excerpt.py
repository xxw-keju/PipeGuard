    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        # Should remain unchanged
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
            else:
                B[i] = T.float32(0)
            if cond[0] > 0:
                C[i] = A[i] * T.float32(2.0)
            else:
                C[i] = T.float32(1)
