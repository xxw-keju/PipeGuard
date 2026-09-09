    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32), B: T.Tensor((4,), T.float32)):
        with T.Kernel(1, threads=1) as _:
            vec = T.Ramp(-2, 1, 4)  # indices: [-2, -1, 0, 1]
            B[T.Ramp(0, 1, 4)] = A[vec]
