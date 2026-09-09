    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 384], T.float32)
            SFQ_tmem = T.alloc_tmem([128, 4], "uint32")
            SFKV0_tmem = T.alloc_tmem([128, 4], "uint32")
            SFKV1_tmem = T.alloc_tmem([128, 4], "uint32")
            T.evaluate(C_tmem[0, 0])
            T.evaluate(SFQ_tmem[0, 0])
            T.evaluate(SFKV0_tmem[0, 0])
            T.evaluate(SFKV1_tmem[0, 0])
