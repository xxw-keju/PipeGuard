    @T.prim_func
    def main(A: T.Tensor((4,), "float16"), B: T.Tensor((4,), "float16")):
        with T.Kernel(1):
            for i in T.grid(4):
                B[i] = A[i] + T.float16(1.5) + T.float16(2.25)
