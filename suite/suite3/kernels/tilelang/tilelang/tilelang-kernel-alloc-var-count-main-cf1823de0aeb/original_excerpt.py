    @T.prim_func
    def main(Out: T.Tensor((32,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            count = T.alloc_var(T.int32, init=0)
            for _ in T.unroll(5):
                count += 1
            Out[tx] = count
