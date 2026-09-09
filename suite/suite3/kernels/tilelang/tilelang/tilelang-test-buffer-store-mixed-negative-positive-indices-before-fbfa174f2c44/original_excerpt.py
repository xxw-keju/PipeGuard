    @T.prim_func
    def before(A: T.Tensor((1024, 512), T.float32)):
        A[-1, 10] = 42.0
