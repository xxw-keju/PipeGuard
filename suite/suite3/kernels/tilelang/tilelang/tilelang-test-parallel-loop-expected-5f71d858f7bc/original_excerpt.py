    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in T.Parallel(128):
                B[i] = A[i]
        else:
            for _i in T.Parallel(128):
                T.evaluate(0)
