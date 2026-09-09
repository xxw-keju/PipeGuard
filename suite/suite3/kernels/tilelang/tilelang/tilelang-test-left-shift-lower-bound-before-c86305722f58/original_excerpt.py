    @T.prim_func
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if T.shift_left(1, i, dtype="int32") >= 1:
                A[i] = 0.0
