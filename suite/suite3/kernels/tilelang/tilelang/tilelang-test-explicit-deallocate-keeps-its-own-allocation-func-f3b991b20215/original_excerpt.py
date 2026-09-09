    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 384], T.float32)
            SF_tmem = T.alloc_tmem([128, 4], "uint32")
            T.evaluate(C_tmem[0, 0])
            T.evaluate(SF_tmem[0, 0])
            T.deallocate_tmem(SF_tmem)
