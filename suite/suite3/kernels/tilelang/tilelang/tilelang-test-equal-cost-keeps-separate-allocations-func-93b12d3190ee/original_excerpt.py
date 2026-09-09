    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            S_tmem = T.alloc_tmem([128, 128], T.float32)
            P_tmem = T.alloc_tmem([128, 128], T.float16)
            O_tmem = T.alloc_tmem([128, 128], T.float32)
            T.evaluate(S_tmem[0, 0])
            T.evaluate(P_tmem[0, 0])
            T.evaluate(O_tmem[0, 0])
