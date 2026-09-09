    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            local = T.alloc_local((32,), T.float32)
            for i, j in T.Parallel(8, 4):
                local[i * 4 + j] = T.float32(1)
