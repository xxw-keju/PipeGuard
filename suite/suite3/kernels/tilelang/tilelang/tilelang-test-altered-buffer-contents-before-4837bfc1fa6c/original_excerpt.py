    @T.prim_func
    def before(A: T.Buffer((1,), "int32"), n: T.int32):
        if A[0] == n:
            A[0] = A[0] + 1
            if A[0] == n:
                A[0] = 0
