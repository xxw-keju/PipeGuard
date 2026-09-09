    @T.prim_func
    def main(Out: T.Tensor((64,), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            counter = T.alloc_var(T.int32, init=7)
            Out[tx] = counter
