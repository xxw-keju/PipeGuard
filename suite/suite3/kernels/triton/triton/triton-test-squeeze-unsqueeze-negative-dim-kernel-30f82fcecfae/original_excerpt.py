    @triton.jit
    def kernel(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.unsqueeze(offset1, -1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.unsqueeze(offset1, -2)
        tl.static_assert(t.shape == [1, N])

        t = tl.unsqueeze(tl.expand_dims(offset1, 0), -1)
        tl.static_assert(t.shape == [1, N, 1])

        t = tl.squeeze(tl.expand_dims(offset1, 1), -1)
        tl.static_assert(t.shape == [N])

        t = tl.squeeze(tl.expand_dims(offset1, 0), -2)
        tl.static_assert(t.shape == [N])
