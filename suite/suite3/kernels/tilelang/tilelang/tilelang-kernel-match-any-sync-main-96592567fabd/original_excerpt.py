    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            value = T.if_then_else(tx < 16, 1, 2)
            peers = T.match_any_sync(value)
            B[tx] = T.cast(peers, "int32")
