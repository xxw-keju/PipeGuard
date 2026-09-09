    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # No changes expected for non-negative indices
        value = A[0]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value
