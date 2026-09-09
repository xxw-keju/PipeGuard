    @T.prim_func
    def kernel(A):
        A: T.Tensor[[32, 32], T.float32]
        B = T.empty([32, 32], T.float32)
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j]
        return B
