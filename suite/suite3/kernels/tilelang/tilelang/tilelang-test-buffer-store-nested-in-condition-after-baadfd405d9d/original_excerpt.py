    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32), flag: T.int32):
        if flag > 0:
            A[1023] = 42.0
        else:
            A[1022] = 24.0
