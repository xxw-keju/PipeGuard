    @T.prim_func
    def expected(flag: T.int32, A: T.Buffer((2,), "float32")):
        if flag > 0:
            A[0] = T.float32(1)
        if flag > 0:
            A[1] = T.float32(2)
