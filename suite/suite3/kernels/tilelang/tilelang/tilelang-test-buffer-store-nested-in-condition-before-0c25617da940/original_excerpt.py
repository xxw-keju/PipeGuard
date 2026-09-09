    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32), flag: T.int32):
        if flag > 0:
            A[-1] = 42.0
        else:
            A[-2] = 24.0
