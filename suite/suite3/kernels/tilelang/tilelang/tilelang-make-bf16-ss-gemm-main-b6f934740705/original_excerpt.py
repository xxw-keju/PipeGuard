    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.bfloat16),
        B: T.Tensor((16, 16), T.bfloat16),
        C: T.Tensor((64, 16), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.bfloat16)
            B_shared = T.alloc_shared((16, 16), T.bfloat16)
            C_local = T.alloc_fragment((64, 16), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C_local, C)
