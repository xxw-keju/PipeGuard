    @T.prim_func
    def main(
        A: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if tx == 0:
                tirx.call_extern("void", "__nanosleep", 100)
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]
