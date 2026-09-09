    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        # Should remain unchanged
        for i in range(128):
            A[i] = T.float32(1.0)
            if A[0] > 0:
                B[i] = A[i]
