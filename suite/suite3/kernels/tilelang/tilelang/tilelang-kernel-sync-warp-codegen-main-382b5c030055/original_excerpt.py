    @T.prim_func
    def main(A: T.Tensor((32,), "float32"), B: T.Tensor((32,), "float32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            A_shared = T.alloc_shared((32,), "float32")
            A_shared[tx] = A[tx]
            T.sync_warp()
            B[tx] = A_shared[tx] * 2.0
