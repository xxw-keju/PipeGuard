        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.bfloat16),
            B: T.Tensor((1024,), dtype=T.bfloat16),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float32)
                T.copy(A, a_frag)
                T.copy(a_frag, B)
