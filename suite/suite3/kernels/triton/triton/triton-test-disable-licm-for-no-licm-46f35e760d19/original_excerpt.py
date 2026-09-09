    @triton.jit
    def for_no_licm(n):
        for i in tl.range(0, n, disable_licm=True):
            print("i", i)
