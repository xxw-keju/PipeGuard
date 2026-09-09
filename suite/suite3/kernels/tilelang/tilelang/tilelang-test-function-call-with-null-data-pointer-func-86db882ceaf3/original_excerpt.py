    @T.prim_func
    def func(A: T.Buffer([16, 16], "int32"), B: T.Buffer([16, 16], "int32")):
        for i, j in T.grid(16, 16):
            B[i, j] = A[i, j]
