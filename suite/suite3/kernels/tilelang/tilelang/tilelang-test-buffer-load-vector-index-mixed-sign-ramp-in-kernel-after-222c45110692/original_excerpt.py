    @T.prim_func
    def after(A: T.Tensor((1024,), T.float32), B: T.Tensor((4,), T.float32)):
        with T.Kernel(1, threads=1) as _:
            vec = T.Ramp(-2, 1, 4)  # noqa: F841
            B[T.Ramp(0, 1, 4)] = A[T.Shuffle([1022, 1023, 0, 1], [0, 1, 2, 3])]
