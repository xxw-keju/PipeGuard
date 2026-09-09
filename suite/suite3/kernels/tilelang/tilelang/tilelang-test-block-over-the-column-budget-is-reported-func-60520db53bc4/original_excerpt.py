    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            first_tmem = T.alloc_tmem([128, 384], T.float32)
            second_tmem = T.alloc_tmem([128, 384], T.float32)
            T.evaluate(first_tmem[0, 0])
            T.evaluate(second_tmem[0, 0])
