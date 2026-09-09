    @triton.jit
    def kernel(ExitEarly, Out, cond: tl.constexpr, mode: tl.constexpr):
        if mode == "dynamic":
            if tl.load(ExitEarly):
                tl.store(Out, 0)
                return
        else:
            if cond:
                tl.store(Out, 0)
                return
        tl.store(Out, 1)
