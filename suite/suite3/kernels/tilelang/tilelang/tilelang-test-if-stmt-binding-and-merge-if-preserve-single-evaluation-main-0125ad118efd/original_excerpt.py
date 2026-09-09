    @T.prim_func
    def main(A: T.Tensor((2,), T.int32)):
        if A[0] > 0:
            A[0] = 0
            A[1] = 1
