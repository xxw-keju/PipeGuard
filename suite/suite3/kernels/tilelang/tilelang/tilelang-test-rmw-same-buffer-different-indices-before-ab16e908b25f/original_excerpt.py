    @T.prim_func
    def before(a: T.Tensor[(64,), T.float8_e4m3fn]):
        for i in T.vectorized(32):
            a[i] = T.cast(
                T.cast(a[i], T.float32) + T.cast(a[i + 32], T.float32),
                T.float8_e4m3fn,
            )
