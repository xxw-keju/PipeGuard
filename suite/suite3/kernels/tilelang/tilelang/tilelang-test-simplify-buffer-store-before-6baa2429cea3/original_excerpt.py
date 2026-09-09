    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = 5
        A[0] = A[0] + 7
