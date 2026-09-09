    @T.prim_func
    def before(A: T.Buffer(1, "int32")):
        A[0] = T.cast(T.ceil(T.log2(T.cast(14, "float64"), dtype="float64"), dtype="float64"), dtype="int32")
