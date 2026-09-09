    @T.prim_func
    def main(
        A: T.Tensor((64,), T.float16),
        B: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.float16)
            for i in T.vectorized(4):
                T.ptx_cp_async(
                    T.access_ptr(S[i], "w", 1),
                    T.access_ptr(A[i], "r", 1),
                    1,
                )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            B[0] = S[0]
