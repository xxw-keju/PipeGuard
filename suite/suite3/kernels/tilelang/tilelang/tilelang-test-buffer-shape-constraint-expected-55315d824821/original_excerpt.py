    @T.prim_func
    def expected(a: T.handle):
        n = T.int64()
        A = T.match_buffer(a, (n * 32,), "float32")
        A[T.int64(0)] = T.float32(0)
