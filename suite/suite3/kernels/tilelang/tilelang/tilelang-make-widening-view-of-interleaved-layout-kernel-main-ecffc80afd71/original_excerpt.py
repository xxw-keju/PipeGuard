    @T.prim_func
    def main():
        with T.Kernel(1, threads=128):
            frag16 = T.alloc_fragment((128, 64), T.bfloat16)
            frag32 = T.alloc_fragment((128, 32), T.float32)
            tmem = T.alloc_tmem((128, 64), T.bfloat16)
            view = T.view(tmem, shape=(128, 32), dtype=T.float32)

            T.annotate_layout({tmem: T.Layout((128, 64), lambda i, j: [i, 2 * j])})
            T.copy(frag16, tmem)
            T.copy(view, frag32)
