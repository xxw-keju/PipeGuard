    @T.prim_func
    def main(
        B: T.Tensor((32,), "int64"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            mask = T.activemask()
            B[tx] = T.cast(mask, "int64")
