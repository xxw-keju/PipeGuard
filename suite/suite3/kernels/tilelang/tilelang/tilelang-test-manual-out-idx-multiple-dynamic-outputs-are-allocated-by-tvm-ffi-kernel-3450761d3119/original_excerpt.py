    @T.prim_func
    def kernel(A, B, C):
        m = T.dynamic("m")
        A: T.Tensor[[m], T.float32]
        B: T.Tensor[[m * 2 + 1], T.float32]
        C: T.Tensor[[m + 1], T.float32]
        with T.Kernel(1, threads=1):
            for i in T.serial(m * 2 + 1):
                B[i] = A[i % m] + 1.0
            for i in T.serial(m + 1):
                C[i] = T.cast(i, T.float32)
