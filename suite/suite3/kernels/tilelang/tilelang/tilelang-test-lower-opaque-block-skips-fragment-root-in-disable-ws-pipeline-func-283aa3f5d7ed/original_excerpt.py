    @T.prim_func
    def func(
        A: T.Tensor((256, 256), T.bfloat16),
        B: T.Tensor((128, 256), T.bfloat16),
        C: T.Tensor((256, 128), T.bfloat16),
    ):
        with T.Kernel(1, threads=256):
            A_shared = T.alloc_shared((256, 128), T.bfloat16)
            B_shared = T.alloc_shared((128, 128), T.bfloat16)
            C_local = T.alloc_fragment((256, 128), T.float32)
            C_shared = T.alloc_shared((256, 128), T.bfloat16)
            T.clear(C_local)
            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[0, k * 128], A_shared)
                T.copy(B[0, k * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C_shared)
            T.copy(C_shared, C[0:256, 0:128])
