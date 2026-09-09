    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.ballot_sync(tx >= 7)
            B[tx] = T.__ffs(mask) - 1
