    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            P_tmem = T.alloc_tmem((128, 128), T.bfloat16)
            V_shared = T.alloc_shared((128, 64), T.bfloat16)
            O_tmem = T.alloc_tmem((128, 64), T.float32)
            done_0 = T.alloc_barrier(1)
            done_1 = T.alloc_barrier(1)
            done_2 = T.alloc_barrier(1)
            done_3 = T.alloc_barrier(1)

            T.gemm(P_tmem[:, 0:32], V_shared[0:32, :], O_tmem, mbar=done_0, clear_accum=True)
            T.gemm(P_tmem[:, 32:64], V_shared[32:64, :], O_tmem, mbar=done_1)
            T.gemm(P_tmem[:, 64:96], V_shared[64:96, :], O_tmem, mbar=done_2)
            T.gemm(P_tmem[:, 96:128], V_shared[96:128, :], O_tmem, mbar=done_3)
