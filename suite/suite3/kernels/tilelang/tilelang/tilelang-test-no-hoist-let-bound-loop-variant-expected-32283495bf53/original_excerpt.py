    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
    ):
        # Should remain unchanged since idx depends on loop variable i
        for i in range(128):
            idx = i % 2
            if idx == 0:
                B[i] = A[i]
