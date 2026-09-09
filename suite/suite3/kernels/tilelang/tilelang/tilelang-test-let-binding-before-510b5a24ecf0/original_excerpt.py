    @T.prim_func
    def before(A: T.Tensor((128, 128), T.float32), B: T.Tensor((128, 128), T.float32)):
        for i in range(128):
            for j in range(128):
                with T.sblock("compute"):
                    factor = T.float32(2.0)
                    value = A[i, j] * factor
                    B[i, j] = value
