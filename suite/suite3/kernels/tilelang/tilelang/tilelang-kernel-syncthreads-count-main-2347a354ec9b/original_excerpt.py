    @T.prim_func
    def main(
        B: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            cnt = T.syncthreads_count(tx < 16)
            B[tx] = cnt
