    @triton.jit
    def while_no_licm(n):
        i = 0
        while tl.condition(i < n, disable_licm=True):
            i = i + 1
            print("i", i)
