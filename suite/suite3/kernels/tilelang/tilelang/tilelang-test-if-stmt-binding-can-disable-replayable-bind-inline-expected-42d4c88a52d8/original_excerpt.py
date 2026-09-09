    @T.prim_func
    def expected(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            bound = T.bind(A[1] + T.float32(2))
            B[0] = bound
            B[1] = bound + T.float32(1)
