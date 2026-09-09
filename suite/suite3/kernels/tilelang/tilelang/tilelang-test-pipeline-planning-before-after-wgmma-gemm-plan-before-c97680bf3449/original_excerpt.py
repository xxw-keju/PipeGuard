    @T.prim_func
    def before(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float16)

            for _i in T.Pipelined(4, num_stages=3):
                T.copy(A[0:64, 0:16], A_shared)
                T.copy(B[0:16, 0:64], B_shared)
                T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
                T.wait_wgmma(0)
                T.copy(C_local, D[0:64, 0:64])
