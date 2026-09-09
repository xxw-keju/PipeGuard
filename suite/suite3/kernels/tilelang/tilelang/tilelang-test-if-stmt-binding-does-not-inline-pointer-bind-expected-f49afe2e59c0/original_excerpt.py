    @T.prim_func(check_well_formed=False)
    def expected(A: T.Buffer((4,), "float32"), C: T.Buffer((4,), "float32")):
        if A[0] >= T.float32(0):
            ptr = T.bind(A.data, type_annotation=A.data)
            B = T.Buffer((4,), "float32", data=ptr)
            C[0] = B[T.int32(1)]
