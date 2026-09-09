    @T.prim_func
    def before(A: T.Buffer((1,), "float32")):
        A[0] = A[0]
