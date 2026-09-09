    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            local = T.alloc_local((32,), T.float32)
            for i in T.Parallel(32):
                local[i] = T.float32(1)
