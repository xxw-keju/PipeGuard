    @T.prim_func
    def kernel(C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            for i in T.Parallel(BC):
                C[bx, i] = s[i] + s[(i + 1) % BC] + s[(i + 2) % BC]
