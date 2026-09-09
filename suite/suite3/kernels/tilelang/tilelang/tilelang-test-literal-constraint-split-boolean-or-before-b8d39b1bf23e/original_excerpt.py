    @T.prim_func
    def before(A: T.Buffer((16, 16), "int32"), n: T.int32):
        for i, j in T.grid(16, 16):
            if i == n or j == n:
                A[i, j] = 0
            else:
                if i == n:
                    A[i, j] = 1
                else:
                    A[i, j] = 2
