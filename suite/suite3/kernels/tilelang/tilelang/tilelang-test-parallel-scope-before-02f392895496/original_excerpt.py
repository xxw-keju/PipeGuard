    @T.prim_func
    def before(A: T.Tensor((128,), T.float32)):
        for i in T.Parallel(128):
            with T.sblock("parallel"):
                value = T.float32(1.0)
                A[i] = value
