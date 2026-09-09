    @T.prim_func
    def expected(A: T.Buffer((16,), "int32")):
        for i in T.serial(16):
            A[i] = i + 1
