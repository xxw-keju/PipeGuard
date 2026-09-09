    @T.prim_func
    def kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float32),
    ):
        with T.Kernel(
            T.ceildiv(N, BLOCK_N),
            T.ceildiv(M, BLOCK_M),
            threads=THREADS,
        ) as (bx, by):
            As = T.alloc_shared((BLOCK_M, BLOCK_K), T.float16)
            Bs = T.alloc_shared((BLOCK_K, BLOCK_N), T.float16)
            Cl = T.alloc_shared((BLOCK_M, BLOCK_N), T.float32)

            T.clear(Cl)
            for k in T.serial(T.ceildiv(K, BLOCK_K)):
                T.copy(A[by * BLOCK_M, k * BLOCK_K], As)
                T.copy(B[k * BLOCK_K, bx * BLOCK_N], Bs)
                T.gemm(As, Bs, Cl)
            T.copy(Cl, C[by * BLOCK_M, bx * BLOCK_N])
