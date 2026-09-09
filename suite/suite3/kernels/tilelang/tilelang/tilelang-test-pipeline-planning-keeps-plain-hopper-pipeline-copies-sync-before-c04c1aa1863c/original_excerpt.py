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
            for k in T.Pipelined(32, num_stages=2):
                T.copy(A[by * 128, k * 32], A_shared)
                T.copy(B[k * 32, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
