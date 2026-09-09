    @T.prim_func
    def expected(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = 1.0
