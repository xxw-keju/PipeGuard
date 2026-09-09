    @T.prim_func
    def before(b: T.Tensor[(16,), T.float8_e4m3fn]):
        a_frag = T.alloc_local((16,), T.float32)
        scale = T.alloc_local((16,), T.float32)
        for i in T.vectorized(16):
            factor = scale[i]
            scaled = a_frag[i] * (factor + T.float32(1))
            b[i] = scaled
