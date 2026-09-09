    @T.prim_func
    def main(B: T.Tensor((4, 4), T.float32), A: T.Tensor((1024,), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for i in T.serial(1024):
                A[i] = T.float32(-1)
            for t in T.serial(4):
                A[T.Ramp(t - 2, 1, 4)] = B[t, T.Ramp(0, 1, 4)]
