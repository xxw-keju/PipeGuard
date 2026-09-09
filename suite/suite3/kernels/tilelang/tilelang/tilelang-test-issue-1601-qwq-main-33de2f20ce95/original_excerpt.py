        @T.prim_func
        def main(
            A: T.Tensor((8,), T.float8_e4m3fn),
        ):
            with T.Kernel(1, threads=32):
                for i in T.vectorized(8):
                    A[i] = 0
