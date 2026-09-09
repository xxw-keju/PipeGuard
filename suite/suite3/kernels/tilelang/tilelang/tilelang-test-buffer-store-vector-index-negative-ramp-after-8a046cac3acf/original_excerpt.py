    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        # vec is unused and can be delimed by Simplify.
        vec = T.Ramp(-4, 1, 4)  # noqa: F841
        values = T.Ramp(0.0, 1.0, 4)
        A[T.Ramp(1020, 1, 4)] = values
