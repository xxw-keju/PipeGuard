    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            iket.mark("float_value", payload=iket.payload(A[0], dtype="float32"))
            A[0] = A[0]
