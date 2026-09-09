    @T.prim_func
    def main(x: T.Tensor((128, 32), T.float32)):
        with T.Kernel(threads=128):
            x_shared = T.alloc_shared((128, 32), T.float32)
            T.copy(x, x_shared, annotations={"prefer_instruction": "tma"})
