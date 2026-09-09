    @T.prim_func
    def before():
        with T.Kernel(8):
            _ = T.launch_thread("threadIdx.x", 128)
            mbarrier = T.alloc_barrier([128, 128, 128, 128])  # noqa: F841
