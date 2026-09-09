    @T.prim_func
    def before(b: T.Tensor[(16,), T.float32]):
        b_frag = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            b[i] = b_frag[i]
