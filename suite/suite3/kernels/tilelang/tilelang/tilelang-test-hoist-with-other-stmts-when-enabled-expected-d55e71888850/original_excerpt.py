    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        if cond[0] > 0:
            for i in range(128):
                C[i] = A[i]
                B[i] = A[i]
        else:
            for i in range(128):
                C[i] = A[i]
                T.evaluate(0)
