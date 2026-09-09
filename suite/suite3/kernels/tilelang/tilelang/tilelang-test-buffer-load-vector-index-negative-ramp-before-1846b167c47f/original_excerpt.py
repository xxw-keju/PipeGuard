    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-4, 1, 4)  # indices: [-4, -3, -2, -1]
        value = A[vec]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value
