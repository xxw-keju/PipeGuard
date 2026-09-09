    @T.prim_func
    def expected(A: T.Buffer((4,), "float32"), B: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            B[0] = A[1] + T.float32(2)
        if A[0] >= T.float32(0):
            B[1] = A[1] + T.float32(2) + T.float32(1)
