    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        A[-1] = 42.0
