    @T.prim_func
    def main(
        A: T.Tensor((4,), T.float16),
        B: T.Tensor((4,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((4,), T.float16)
            T.async_copy(A[0:4], S)
            T.copy(S, B[0:4])
