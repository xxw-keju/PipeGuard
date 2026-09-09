    @T.prim_func
    def kernel_with_tuple_kernel_binding():
        with T.Kernel(1, threads=32) as (pid,):
            print(pid)
            pass
