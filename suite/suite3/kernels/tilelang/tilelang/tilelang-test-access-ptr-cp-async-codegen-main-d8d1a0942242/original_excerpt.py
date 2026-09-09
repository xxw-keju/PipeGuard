    @T.prim_func
    def main(
        A: T.Tensor((64,), T.uint8),
        B: T.Tensor((64,), T.uint8),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((64,), T.uint8)
            T.ptx_cp_async(
                T.access_ptr(S[8], "w", 16),
                T.access_ptr(A[16], "r", 16),
                16,
            )
            # Keep the shared buffer live so the pointers remain in generated code.
            B[0] = S[8]
