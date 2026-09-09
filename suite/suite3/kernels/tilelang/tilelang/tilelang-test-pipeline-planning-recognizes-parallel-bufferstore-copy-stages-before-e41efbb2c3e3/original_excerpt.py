    @T.prim_func
    def before(
        A: T.Tensor((1024, 32), T.float32),
        B: T.Tensor((32, 1024), T.float32),
        C: T.Tensor((1024, 1024), T.float32),
    ):
        with T.Kernel(8, 8, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 32), T.float32)
            B_shared = T.alloc_shared((32, 128), T.float32)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(32, num_stages=3):
                for i, k in T.Parallel(128, 32):
                    A_shared[i, k] = A[by * 128 + i, ko * 32 + k]
                for k, j in T.Parallel(32, 128):
                    B_shared[k, j] = B[ko * 32 + k, bx * 128 + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * 128, bx * 128])
