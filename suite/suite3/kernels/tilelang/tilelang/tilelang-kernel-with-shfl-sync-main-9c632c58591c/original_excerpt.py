    @T.prim_func
    def main(
        A: T.Tensor((32,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            val = tx * 10
            broadcast = T.shfl_sync(val, 31)
            A[tx] = broadcast
