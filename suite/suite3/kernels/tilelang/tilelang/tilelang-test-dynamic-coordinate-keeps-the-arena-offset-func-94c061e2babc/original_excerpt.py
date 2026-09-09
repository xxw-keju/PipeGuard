    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 384], T.float32)
            SF_tmem = T.alloc_tmem([128, 8], "uint32")
            stage = T.alloc_var(T.int32, init=0)
            T.evaluate(SF_tmem[0, stage * 4])
