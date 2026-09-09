    @T.prim_func
    def expected(A: T.Buffer(1, "bool"), i: T.int32):
        if T.floormod(i, -2) == 0:
            A[0] = True
