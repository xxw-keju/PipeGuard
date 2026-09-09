    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Broadcast(-1, 4)  # noqa: F841
        values = T.Broadcast(42.0, 4)
        A[T.Broadcast(1023, 4)] = values
