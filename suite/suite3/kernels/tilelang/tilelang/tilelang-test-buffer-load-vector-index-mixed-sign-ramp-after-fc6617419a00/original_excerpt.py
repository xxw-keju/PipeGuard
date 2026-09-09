    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-2, 1, 4)  # noqa: F841
        value = A[T.Shuffle([1022, 1023, 0, 1], [0, 1, 2, 3])]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value
