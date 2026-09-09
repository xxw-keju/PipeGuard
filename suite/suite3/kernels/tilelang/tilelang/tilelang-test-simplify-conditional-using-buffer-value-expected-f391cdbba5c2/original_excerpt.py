    @T.prim_func
    def expected(A: T.Buffer(1, "int32")):
        A[0] = 0
        A[0] = 42
