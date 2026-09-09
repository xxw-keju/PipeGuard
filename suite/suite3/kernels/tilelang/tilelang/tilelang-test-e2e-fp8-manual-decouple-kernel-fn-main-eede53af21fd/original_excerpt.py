        @T.prim_func
        def main(
            A: T.Tensor((1024,), dtype=T.float8_e4m3fn),
            B: T.Tensor((1024,), dtype=T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=64):
                a_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                b_frag = T.alloc_fragment((1024,), dtype=T.float32)
                c_frag = T.alloc_fragment((1024,), dtype=T.float8_e4m3fn)
                T.copy(A, a_frag)
                T.copy(a_frag, b_frag)
                T.copy(b_frag, c_frag)
                T.copy(c_frag, B)
