    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=128) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            T.clear(s)
            for i in T.Parallel(BC):
                s[i] = s[i] + A[bx, i]
            for i in T.Parallel(BC):
                C[bx, i] = s[i]
