    @T.prim_func
    def main(Out: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            a = T.alloc_var(T.int32, init=3)
            b = T.alloc_var(T.int32, init=4)
            Out[tx] = a + b
