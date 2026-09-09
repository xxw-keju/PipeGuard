    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in range(128):
                B[i] = A[i]
                C[i] = A[i] * T.float32(2.0)
        else:
            for _i in range(128):
                T.evaluate(0)
                T.evaluate(0)
