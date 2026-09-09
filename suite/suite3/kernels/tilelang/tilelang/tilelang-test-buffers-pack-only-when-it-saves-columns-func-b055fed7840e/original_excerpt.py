    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            C_tmem = T.alloc_tmem([128, 128], T.float32)
            SFA_tmem = T.alloc_tmem([128, 4], "uint32")
            SFB_tmem = T.alloc_tmem([128, 4], "uint32")
            T.evaluate(C_tmem[0, 0])
            T.evaluate(SFA_tmem[0, 0])
            T.evaluate(SFB_tmem[0, 0])
