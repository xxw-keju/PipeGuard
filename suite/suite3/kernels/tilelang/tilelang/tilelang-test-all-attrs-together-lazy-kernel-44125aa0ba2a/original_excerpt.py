    @T.prim_func
    def kernel(A):
        A: T.Tensor[[64, 64], T.float32]
        T.annotate_pass_configs({PassConfigKey.TL_ENABLE_FAST_MATH: True})
        T.annotate_compile_flags(["--use_fast_math"])
        B = T.empty([64, 64], T.float32)
        with T.Kernel(1):
            for i in T.serial(64):
                for j in T.serial(64):
                    B[i, j] = A[i, j] * 2.0
        return B
