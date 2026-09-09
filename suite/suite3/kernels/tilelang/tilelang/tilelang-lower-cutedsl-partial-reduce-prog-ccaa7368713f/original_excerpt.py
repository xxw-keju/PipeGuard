    @T.prim_func
    def prog(A: T.Tensor((1, 512), "float32"), B: T.Tensor((1,), "float32")):
        with T.Kernel(1, threads=128):
            x_frag = T.alloc_fragment((1, 512), "float32")
            sum_frag = T.alloc_fragment((1,), "float32")
            T.annotate_layout(
                {
                    x_frag: T.Fragment(x_frag.shape, forward_fn=lambda i, j: (j // 8, j % 8)),
                    sum_frag: T.Fragment(sum_frag.shape, forward_fn=lambda i, rep: (rep, 0), replicate=64),
                }
            )
            for i, j in T.Parallel(1, 512):
                x_frag[i, j] = A[i, j]
            T.reduce_sum(x_frag, sum_frag, dim=1)
            for i in T.Parallel(1):
                B[i] = sum_frag[i]
