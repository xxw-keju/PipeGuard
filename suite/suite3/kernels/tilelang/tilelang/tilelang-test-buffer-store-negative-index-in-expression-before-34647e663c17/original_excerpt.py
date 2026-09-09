    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        for i in T.serial(1, 1024):
            A[-i] = i * 2.0
