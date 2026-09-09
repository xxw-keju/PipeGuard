    @T.prim_func
    def before(a: T.Tensor[(256,), T.bfloat16], b: T.Tensor[(256,), T.bfloat16]):
        for i in T.vectorized(8):
            T.ptx_cp_async(
                T.address_of(b[i]),
                T.address_of(a[i]),
                1,
                T.cast(i, "int64") < T.int64(128),
            )
