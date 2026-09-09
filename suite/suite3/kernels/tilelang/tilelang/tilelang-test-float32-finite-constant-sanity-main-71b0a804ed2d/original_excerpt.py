    @T.prim_func
    def main(A: T.Tensor((1,), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1):
            B[0] = A[0] + T.cast(1.5, "float32") + T.cast(2.25, "float32")
