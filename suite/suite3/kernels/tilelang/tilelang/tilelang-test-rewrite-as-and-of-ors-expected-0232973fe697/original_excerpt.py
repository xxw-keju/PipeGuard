    @T.prim_func
    def expected(A: T.Buffer(3, "bool")):
        T.evaluate((A[0] or A[1]) and (A[0] or A[2]))
