    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            A_tmem = T.alloc_tmem([128, 128], T.float32)
            B_tmem = T.alloc_tmem([128, 64], T.float32)
            T.deallocate_tmem(A_tmem)
