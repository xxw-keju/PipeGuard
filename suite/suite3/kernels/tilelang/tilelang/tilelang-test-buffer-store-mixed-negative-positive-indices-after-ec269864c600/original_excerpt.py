    @T.prim_func
    def after(A: T.Tensor((1024, 512), T.float32)):
        A[1023, 10] = 42.0
