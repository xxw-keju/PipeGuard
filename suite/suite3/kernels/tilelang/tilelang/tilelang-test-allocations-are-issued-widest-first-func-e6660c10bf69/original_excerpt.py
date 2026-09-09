    @T.prim_func
    def func():
        with T.Kernel(1, threads=128):
            narrow_tmem = T.alloc_tmem([128, 32], T.float32)
            wide_tmem = T.alloc_tmem([128, 256], T.float32)
            T.evaluate(narrow_tmem[0, 0])
            T.evaluate(wide_tmem[0, 0])
