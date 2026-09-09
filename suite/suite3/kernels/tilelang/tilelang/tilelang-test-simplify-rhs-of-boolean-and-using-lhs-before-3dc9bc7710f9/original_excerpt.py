    @T.prim_func
    def before(A: T.Buffer(1, "bool"), n: T.int32):
        A[0] = n < 5 and n < 10
