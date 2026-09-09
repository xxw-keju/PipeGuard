    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S0 = T.alloc_shared((4,), T.float16)
            S1 = T.alloc_shared((4,), T.float16)
            T.copy(A[0:4], S0)
            # shared->shared cannot use cp.async and should fail for async_copy.
            T.async_copy(S0, S1)
            T.copy(S1, B[0:4])
