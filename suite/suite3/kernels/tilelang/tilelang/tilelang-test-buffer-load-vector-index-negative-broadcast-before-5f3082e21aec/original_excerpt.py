    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Broadcast(-1, 4)
        value = A[vec]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value
