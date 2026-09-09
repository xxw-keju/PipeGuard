    @T.prim_func
    def main(A: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.sync_grid()
            A[tx] = T.float32(tx)
