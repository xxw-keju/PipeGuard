    @T.prim_func
    def empty_kernel():
        with T.Kernel(1, threads=32) as thread_idx:
            pass
