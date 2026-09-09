    @T.prim_func
    def expected(A: T.Tensor((1024,), T.float32), B: T.Tensor((4, 4), T.float32)):
        with T.Kernel(1, threads=1) as _:
            for t in T.serial(4):
                B[t, T.Ramp(0, 1, 4)] = A[
                    T.Shuffle(
                        [
                            T.Select(t < 2, t + 1022, t - 2),
                            T.Select(t < 1, t + 1023, t - 1),
                            t,
                            t + 1,
                        ],
                        [0, 1, 2, 3],
                    )
                ]
