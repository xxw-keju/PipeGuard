    @T.prim_func
    def shared_kernel():
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared_mem = T.alloc_shared((32), dtype="float32", scope="shared")
            if tx % 2 == 0:
                a = shared_mem[tx]
                shared_mem[tx ^ 1] = a
