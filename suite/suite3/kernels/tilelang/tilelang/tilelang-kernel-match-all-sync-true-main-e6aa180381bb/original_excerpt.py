    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            result = T.match_all_sync(7)
            B[tx] = T.cast(result, "int32")
