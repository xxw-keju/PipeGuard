    @T.prim_func
    def before(a: T.handle):
        n = T.int64()
        A = T.match_buffer(a, (n * 32,), "float32")
        A[T.min(T.int64(0), n)] = T.float32(0)
