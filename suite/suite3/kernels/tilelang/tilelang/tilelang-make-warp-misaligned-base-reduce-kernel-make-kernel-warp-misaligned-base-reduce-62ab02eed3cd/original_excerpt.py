        @T.prim_func
        def warp_misaligned_base_reduce(
            x: T.Tensor((1, 512), "float32"),
            out: T.Tensor((1,), "float32"),
        ) -> None:
            with T.Kernel(1, threads=128):
                x_frag = T.alloc_fragment((1, 512), "float32")
                sum_frag = T.alloc_fragment((1,), "float32")
                T.annotate_layout(
                    {
                        x_frag: T.Fragment(
                            x_frag.shape,
                            forward_fn=lambda i, j: (16 + j // 8, j % 8),
                        ),
                        sum_frag: T.Fragment(
                            sum_frag.shape,
                            forward_fn=lambda i, rep: (16 + rep, 0),
                            replicate=64,
                        ),
                    }
                )
                for i, j in T.Parallel(1, 512):
                    x_frag[i, j] = x[i, j]
                T.reduce_sum(x_frag, sum_frag, dim=1)
                for i in T.Parallel(1):
                    out[i] = sum_frag[i]
