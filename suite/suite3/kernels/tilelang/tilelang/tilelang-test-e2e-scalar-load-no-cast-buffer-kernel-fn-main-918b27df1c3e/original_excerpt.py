        @T.prim_func
        def main(
            a: T.Tensor[(32,), T.float8_e4m3fn],
            b: T.Tensor[(1,), T.float32],
            c: T.Tensor[(32,), T.float8_e4m3fn],
        ):
            with T.Kernel(1, threads=32):
                a_local = T.alloc_local((32,), T.float8_e4m3fn)
                T.copy(a, a_local)

                for i in T.vectorized(32):
                    c[i] = a_local[i] * b[0]
