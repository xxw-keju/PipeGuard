    @T.prim_func
    def main(dst: T.StridedTensor((M, N), (PITCH, 1), "int32")):
        with T.Kernel(1, threads=1):
            for row in T.serial(M):
                T.stg32(dst[row, 0], T.Cast("uint32", row + 1))
