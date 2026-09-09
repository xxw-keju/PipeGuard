    @triton.jit
    def kernel():
        a = tl.full((64, ), 0, tl.uint32)
        b = tl.full((64, ), 1, tl.float32)
        c = tl.full((64, ), 2, tl.float32)
        tl.where(a, b, c)
