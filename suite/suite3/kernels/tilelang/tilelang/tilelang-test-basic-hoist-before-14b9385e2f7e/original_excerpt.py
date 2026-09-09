    @T.prim_func
    def before(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        for i in range(128):
            if cond[0] > 0:
                B[i] = A[i]
