    @T.prim_func
    def main(
        A: T.Tensor((64, 32), T.float16),
        B: T.Tensor((64, 32), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 32), T.float16)
            B_shared = T.alloc_shared((64, 32), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_local)
            T.wgmma_gemm(
                A_shared[:, 16:],
                B_shared[:, 16:],
                C_local,
                transpose_B=True,
            )
            T.wait_wgmma(0)
            T.copy(C_local, D)
