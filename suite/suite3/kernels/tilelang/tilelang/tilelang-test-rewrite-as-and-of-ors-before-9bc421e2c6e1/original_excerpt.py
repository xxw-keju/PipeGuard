    @T.prim_func
    def before(A: T.Buffer(3, "bool")):
        T.evaluate(A[0] or (A[1] and A[2]))
