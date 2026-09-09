    @T.prim_func
    def before(A: T.Buffer(1, "bool"), i: T.int32):
        if T.floormod(0 - i, 2) == 0:
            A[0] = T.floormod(i, 2) == 0
