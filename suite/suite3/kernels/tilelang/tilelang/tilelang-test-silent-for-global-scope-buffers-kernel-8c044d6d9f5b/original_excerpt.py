    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            for i in T.Parallel(BC):
                C[bx, i] = A[bx, i]
