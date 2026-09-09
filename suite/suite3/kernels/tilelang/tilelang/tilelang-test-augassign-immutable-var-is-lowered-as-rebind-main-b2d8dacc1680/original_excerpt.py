    @T.prim_func
    def main(
        A: T.Tensor[(1,), T.int32],
        B: T.Tensor[(1,), T.int32],
    ):
        x = A[0]
        x -= 1
        B[0] = x
