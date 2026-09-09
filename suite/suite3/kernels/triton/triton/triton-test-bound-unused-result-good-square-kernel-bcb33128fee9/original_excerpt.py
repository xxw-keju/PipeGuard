    @triton.jit
    def good_square_kernel():
        a = Square(tl.full((64, 64), 0.0, tl.float32))
        a = a.compute()
