    @T.prim_func
    def after(A: T.Tensor((T.int64(3000000000),), T.float32)):
        vec = T.Ramp(T.int32(-2), T.int32(1), 4)  # noqa: F841
        value = A[
            T.Shuffle(
                [
                    T.int64(2999999998),
                    T.int64(2999999999),
                    T.int64(0),
                    T.int64(1),
                ],
                [0, 1, 2, 3],
            )
        ]
        B = T.alloc_buffer((4,), T.float32)
        B[T.Ramp(0, 1, 4)] = value
