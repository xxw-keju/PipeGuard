    @triton.jit
    def expand_dims_kernel(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.expand_dims(offset1, 0)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, 1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.expand_dims(offset1, -2)
        tl.static_assert(t.shape == [1, N])

        t = tl.expand_dims(offset1, (0, -1))
        tl.static_assert(t.shape == [1, N, 1])

        t = tl.expand_dims(offset1, (0, 1, 3))
        tl.static_assert(t.shape == [1, 1, N, 1])

        t = tl.expand_dims(offset1, (-4, 2, -1))
        tl.static_assert(t.shape == [1, N, 1, 1])

        t = tl.expand_dims(offset1, (3, 1, 2))
        tl.static_assert(t.shape == [N, 1, 1, 1])

        scalar = tl.sum(offset1)
        tl.static_assert(scalar.shape == [])
        t = tl.expand_dims(scalar, 0)
        tl.static_assert(t.shape == [1])

        t = tl.expand_dims(scalar, -1)
        tl.static_assert(t.shape == [1])

        # N is a scalar that's not even a tl.tensor -- this should work too.
        t = tl.expand_dims(N, -1)
        tl.static_assert(t.shape == [1])
