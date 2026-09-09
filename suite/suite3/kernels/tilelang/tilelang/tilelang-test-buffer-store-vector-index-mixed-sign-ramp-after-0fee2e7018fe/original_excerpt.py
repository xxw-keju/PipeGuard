    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-2, 1, 4)  # noqa: F841
        values = T.Ramp(0.0, 1.0, 4)
        A[T.Shuffle([1022, 1023, 0, 1], [0, 1, 2, 3])] = values
