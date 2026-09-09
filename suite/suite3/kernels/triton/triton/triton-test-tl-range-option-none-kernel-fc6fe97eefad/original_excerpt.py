    @triton.jit
    def kernel(ub):
        for i in tl.range(0, ub, num_stages=None, loop_unroll_factor=None):
            print("i", i)
