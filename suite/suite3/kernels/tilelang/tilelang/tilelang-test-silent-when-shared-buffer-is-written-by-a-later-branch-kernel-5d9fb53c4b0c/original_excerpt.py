    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=128) as bx:
            s = T.alloc_shared((BC,), torch.float32)
            tx = T.get_thread_binding()
            if tx < 64:
                for i in T.Parallel(BC):
                    C[bx, i] = s[i]
            else:
                T.copy(A[bx, :], s)
