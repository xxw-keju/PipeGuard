    @T.prim_func
    def expected(A: T.Buffer((16,), "int32"), n: T.int32):
        for i in T.serial(16):
            if i == n:
                A[i] = 1
