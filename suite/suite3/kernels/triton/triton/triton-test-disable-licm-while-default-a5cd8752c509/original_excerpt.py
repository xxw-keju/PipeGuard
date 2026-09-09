    @triton.jit
    def while_default(n):
        i = 0
        while tl.condition(i < n):
            i = i + 1
            print("i", i)
