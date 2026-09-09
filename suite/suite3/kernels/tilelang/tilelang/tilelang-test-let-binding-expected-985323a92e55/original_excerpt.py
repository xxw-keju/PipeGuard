    @T.prim_func
    def expected(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.sblock("compute"):
                    B[i, j] = A[i, j] * T.float32(2.0)
