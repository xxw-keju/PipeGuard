    @T.prim_func
    def kernel(
        A: T.Tensor((32,), T.float16),
        B: T.Tensor((32,), T.float16),
        C: T.Tensor((32,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            A_shared = T.alloc_shared((32,), T.float16)
            B_shared = T.alloc_shared((32,), T.float16)
            A_shared[0] = A[0]
            B_shared[0] = B[0]
            T.tvm_storage_sync("shared")
            C[0] = A_shared[0] + B_shared[0]
