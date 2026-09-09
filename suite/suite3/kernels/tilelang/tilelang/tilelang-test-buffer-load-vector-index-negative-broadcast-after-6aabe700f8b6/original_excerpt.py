    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Broadcast(-1, 4)  # noqa: F841
        value = A[T.Broadcast(1023, 4)]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value
