    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        B = T.alloc_buffer((1024,), T.float32)
        for i in T.serial(1, 1024):
            value = A[1024 - i]
            B[1024 - i] = value
