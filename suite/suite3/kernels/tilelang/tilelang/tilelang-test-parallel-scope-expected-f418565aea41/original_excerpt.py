    @T.prim_func
    def expected(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.sblock("parallel"):
                A[i] = T.float32(1.0)
