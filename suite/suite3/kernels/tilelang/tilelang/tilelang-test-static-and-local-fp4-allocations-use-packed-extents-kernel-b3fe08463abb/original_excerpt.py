    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=1):
            A_shared = T.alloc_shared((127,), T.float4_e2m1fn, scope="shared")
            A_local = T.alloc_local((127,), T.float4_e2m1fn)
            A_shared[126] = T.cast(0.0, T.float4_e2m1fn)
            A_local[126] = A_shared[126]
