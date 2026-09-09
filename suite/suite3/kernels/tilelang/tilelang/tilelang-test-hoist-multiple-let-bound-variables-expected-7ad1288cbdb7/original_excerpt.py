    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((2,), T.float32),
    ):
        # Let bindings are hoisted before the if, redundant inner LetStmts are removed
        x = C[0]
        y = C[1]
        if x + y >= T.float32(0):
            for i in range(128):
                B[i] = A[i]
        else:
            for _i in range(128):
                T.evaluate(0)
