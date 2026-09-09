    @triton.jit
    def good_cube_kernel():
        a = tl.full((64, 64), 0.0, tl.float32)
        a = cube(a)
