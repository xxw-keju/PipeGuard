    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float64)):
        with T.Kernel(1):
            value = T.alloc_var(T.float64, init=1e300)
            A[0] = value
