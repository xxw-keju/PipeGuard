    @T.prim_func
    def main(A: T.Tensor((8,), "float32"), B: T.Tensor((8,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(8):
                B[i] = A[i] * 2.0
