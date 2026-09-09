    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            B_shared = T.alloc_shared((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)
            B_shared[126] = T.cast(1.0, T.float4_e2m1fn)
            T.tvm_storage_sync("shared")
            A_shared[0] = B_shared[126]
