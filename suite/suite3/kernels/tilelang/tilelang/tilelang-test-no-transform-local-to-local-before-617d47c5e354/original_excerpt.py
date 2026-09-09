    @T.prim_func
    def before():
        a_frag = T.alloc_local((16,), T.float32)
        b_frag = T.alloc_local((16,), T.float4_e2m1fn)
        for i in T.vectorized(16):
            b_frag[i] = a_frag[i]
