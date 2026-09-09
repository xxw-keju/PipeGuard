    @T.prim_func
    def main(a: T.Tensor[(1,), T.int32], n: T.int32):
        with T.Kernel(1, threads=128):
            if n % 4 == 0:
                T.assume(n > 0)
                a[0] = n
