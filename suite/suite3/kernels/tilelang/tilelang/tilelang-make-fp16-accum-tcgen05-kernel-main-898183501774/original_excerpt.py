    @T.prim_func
    def main(
        A: T.Tensor((128, 128), T.float8_e4m3fn),
        B: T.Tensor((128, 128), T.float8_e4m3fn),
        D: T.Tensor((128, 128), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            B_shared = T.alloc_shared((128, 128), T.float8_e4m3fn)
            C_tmem = T.alloc_tmem((128, 128), T.float16)
            mbar = T.alloc_barrier(1)
            C_local = T.alloc_fragment((128, 128), T.float16)
            C_shared = T.alloc_shared((128, 128), T.float16)

            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_tmem, transpose_B=True, mbar=mbar, clear_accum=True)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(C_tmem, C_local)
            T.copy(C_local, C_shared)
            T.copy(C_shared, D)
