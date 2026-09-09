    @T.prim_func
    def main(a: T.Tensor[(1,), T.int32]):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            T.assume(tx < 128)
            if tx == 0:
                a[0] = tx
