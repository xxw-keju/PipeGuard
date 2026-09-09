    @T.prim_func
    def kernel(A, B):
        A: T.Tensor[[32, 32], T.float32]
        B: T.Tensor[[32, 32], T.float32]
        T.annotate_compile_flags(["--use_fast_math"])
        with T.Kernel(1):
            for i in T.serial(32):
                for j in T.serial(32):
                    B[i, j] = A[i, j] + 1.0
