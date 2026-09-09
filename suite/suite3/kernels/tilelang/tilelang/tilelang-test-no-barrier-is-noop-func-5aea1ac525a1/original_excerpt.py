    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            buf = T.alloc_shared((16,), T.float16)
            buf[0] = T.float16(0)
