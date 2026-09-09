    @T.prim_func
    def main(
        A: T.Tensor((32,), T.float32),
        B: T.Tensor((32,), T.float32),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            B[tx] = T.fast_rcp(A[tx])
