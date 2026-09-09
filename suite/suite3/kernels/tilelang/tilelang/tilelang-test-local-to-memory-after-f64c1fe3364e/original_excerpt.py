    @T.prim_func
    def after(b: T.Tensor[(16,), T.float4_e2m1fn]):
        b_frag = T.alloc_local((16,), T.float32)
        with T.sblock("decoupled_cast"):
            T.sblock_attr({"lexical_alloc_scope": 1})
            T.reads()
            T.writes()
            b_local_cast = T.decl_buffer((16,), T.float4_e2m1fn, scope="local")
            for i in T.vectorized(16):
                b_local_cast[i] = T.cast(b_frag[i], T.float4_e2m1fn)
            for i_copy in T.vectorized(16):
                b[i_copy] = b_local_cast[i_copy]
