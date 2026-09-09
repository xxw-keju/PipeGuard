    @T.prim_func
    def kernel(
        A: T.Tensor((4,), "int8"),
        B: T.Tensor((4,), "int8"),
        C: T.Tensor((1,), "int32"),
    ):
        T.dp4a(A[0], B[0], C[0])
