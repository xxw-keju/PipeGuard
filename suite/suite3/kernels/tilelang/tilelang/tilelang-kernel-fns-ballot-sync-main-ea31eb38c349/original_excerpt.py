    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.cast(T.ballot_sync(tx >= 7), "uint32")
            B[tx] = T.cast(T.__fns(mask, 0, 3), "int32")
