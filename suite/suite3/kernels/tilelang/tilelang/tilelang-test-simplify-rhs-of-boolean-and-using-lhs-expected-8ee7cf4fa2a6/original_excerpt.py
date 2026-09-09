    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5
