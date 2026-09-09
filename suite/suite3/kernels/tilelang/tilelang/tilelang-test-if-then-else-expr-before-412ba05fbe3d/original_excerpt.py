    @T.prim_func
    def before(A: T.Buffer(16, "float32")):
        for i in T.serial(16):
            if i < 12:
                A[i] = T.if_then_else(i < 12, 1.0, 2.0, dtype="float32")
