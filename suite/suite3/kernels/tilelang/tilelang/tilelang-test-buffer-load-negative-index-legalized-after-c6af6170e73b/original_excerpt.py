    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        value = A[1023]  # A[-1] becomes A[1023]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value
