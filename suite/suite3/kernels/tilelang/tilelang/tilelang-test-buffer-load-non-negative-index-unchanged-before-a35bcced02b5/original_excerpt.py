    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        value = A[0]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value
