    @T.prim_func
    def prog(A: T.Tensor((16,), "uint8"), B: T.Tensor((16,), "uint8")):
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((16,), "uint8", scope="shared")
            T.ptx_cp_async(T.access_ptr(A_shared[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
            B[0] = A_shared[0]
