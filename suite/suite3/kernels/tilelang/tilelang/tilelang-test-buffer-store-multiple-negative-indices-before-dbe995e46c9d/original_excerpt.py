    @T.prim_func
    def before(A: T.Tensor((1024, 512, 256), T.float32)):
        A[-1, -2, -3] = 42.0
