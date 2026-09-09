    @T.prim_func
    def before(A: T.Tensor((T.int64(3000000000),), T.float32)):
        vec = T.Ramp(T.int32(-2), T.int32(1), 4)
        value = A[vec]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value
