    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        i = T.Var("i", T.int32)
        value = A[i]
        B = T.alloc_buffer((1,), T.float32)
        B[0] = value
