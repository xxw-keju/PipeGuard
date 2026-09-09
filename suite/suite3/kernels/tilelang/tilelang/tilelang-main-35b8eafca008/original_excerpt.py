@T.prim_func
def main(A: T.Tensor((M, K), T.float16), B: T.Tensor((K, N), T.float16), C: T.Tensor((M, N), T.float32)):
    with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=128) as (bx, by):
        As = T.alloc_shared((BM, BK), "float16")
        Bs = T.alloc_shared((BK, BN), "float16")
        Cl = T.alloc_fragment((BM, BN), "float32")
        T.clear(Cl)
        for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
            T.copy(A[by * BM, k * BK], As)
            T.copy(B[k * BK, bx * BN], Bs)
            T.gemm(As, Bs, Cl)
        T.copy(Cl, C[by * BM, bx * BN])
