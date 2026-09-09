    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 16), T.float16),
        C: T.Tensor((64, 16), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 16), T.float16)
            P = T.alloc_fragment((64, 16), T.float32)
            P_half = T.alloc_fragment((64, 16), T.float16)
            C_local = T.alloc_fragment((64, 16), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)

            for i, j in T.Parallel(64, 16):
                P[i, j] = T.cast(A_shared[i, j], T.float32)

            T.copy(P, P_half)
            T.clear(C_local)
            T.gemm(P_half, B_shared, C_local, policy=T.GemmWarpPolicy.FullRow)
            T.copy(C_local, C)
