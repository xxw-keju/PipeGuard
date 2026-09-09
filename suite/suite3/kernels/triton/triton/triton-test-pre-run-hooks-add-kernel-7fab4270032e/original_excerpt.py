    @triton.jit
    def add_kernel(a_ptr, n_elements: tl.constexpr):
        offsets = tl.arange(0, n_elements)
        a = tl.load(a_ptr + offsets)
        a += 2
        tl.store(a_ptr + offsets, a)
