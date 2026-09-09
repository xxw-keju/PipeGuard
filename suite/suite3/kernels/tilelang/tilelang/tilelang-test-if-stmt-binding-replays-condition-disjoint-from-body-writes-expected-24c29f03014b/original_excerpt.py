    @T.prim_func
    def expected(A: T.Buffer((1,), "float32"), B: T.Buffer((2,), "float32")):
        if A[0] >= T.float32(0):
            B[0] = T.float32(1)
        if A[0] >= T.float32(0):
            B[1] = T.float32(2)
