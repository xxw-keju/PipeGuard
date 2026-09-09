    @T.prim_func
    def expected(
        A: T.Tensor((16, 128), T.float32),
        B: T.Tensor((16, 128), T.float32),
        cond: T.Tensor((1,), T.int32),
    ):
        # if is hoisted outside both loops since cond[0] is invariant to both
        if cond[0] > 0:
            for i in range(16):
                for j in range(128):
                    B[i, j] = A[i, j]
        else:
            for _i in range(16):
                for _j in range(128):
                    T.evaluate(0)
