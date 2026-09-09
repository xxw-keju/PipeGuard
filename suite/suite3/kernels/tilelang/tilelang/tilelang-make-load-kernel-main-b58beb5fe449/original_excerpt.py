    @T.prim_func
    def main(
        src: T.StridedTensor((M, N), (PITCH, 1), "int32"),
        out: T.Tensor((M,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for row in T.serial(M):
                out[row] = T.reinterpret(T.ldg32(src[row, 0]), "int32")
