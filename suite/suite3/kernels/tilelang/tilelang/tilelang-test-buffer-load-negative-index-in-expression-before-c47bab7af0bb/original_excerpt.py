    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        B = T.alloc_buffer((1024,), T.float32)
        for i in T.serial(1, 1024):
            value = A[-i]
            B[-i] = value
