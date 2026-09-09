    @T.prim_func
    def main(A: T.Tensor((128, 4), T.float), B: T.Tensor((128, 4), T.float)):
        with T.Kernel(1, threads=128) as _:
            A_frag = T.alloc_fragment((128, 4), T.float)
            B_frag = T.alloc_fragment((128, 4), T.float)
            S_frag = T.alloc_fragment((4,), T.float)
            T.annotate_layout(
                {
                    A_frag: T.Fragment(A_frag.shape, lambda i, j: (i, j)),
                }
            )
            for i, j in T.Parallel(128, 4):
                A_frag[i, j] = S_frag[j]
            for i, j in T.Parallel(128, 4):
                B_frag[i, j] = S_frag[j]
