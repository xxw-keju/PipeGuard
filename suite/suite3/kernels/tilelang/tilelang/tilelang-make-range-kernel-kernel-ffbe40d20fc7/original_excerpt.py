    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32), iket.range("compute_phase"):
            iket.mark("compute_tick")
            A[0] = A[0]
