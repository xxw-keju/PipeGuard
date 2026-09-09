    @T.prim_func
    def before(A: T.Tensor((1024, 512, 256), T.float32)):
        value = A[-1, -2, -3]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value
