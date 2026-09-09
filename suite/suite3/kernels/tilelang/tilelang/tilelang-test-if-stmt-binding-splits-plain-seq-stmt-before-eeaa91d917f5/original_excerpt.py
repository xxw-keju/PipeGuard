    @T.prim_func
    def before(A: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            A[0] = T.float32(1)
            A[1] = T.float32(2)
