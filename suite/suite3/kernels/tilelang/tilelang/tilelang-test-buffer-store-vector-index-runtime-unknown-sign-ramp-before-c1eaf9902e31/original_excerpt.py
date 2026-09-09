    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for t in T.serial(4):
                A[T.Ramp(t - 2, 1, 4)] = B[t, T.Ramp(0, 1, 4)]
