    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        for i in T.serial(1, 1024):
            A[1024 - i] = i * 2.0
