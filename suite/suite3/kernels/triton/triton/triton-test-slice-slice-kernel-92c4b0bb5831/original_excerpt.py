    @triton.jit
    def slice_kernel(XBLOCK: tl.constexpr):
        data = tl.arange(0, XBLOCK)
        tl.static_assert(data.shape == [XBLOCK])

        t = data[None, :]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, None:]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, :None]
        tl.static_assert(t.shape == [1, XBLOCK])

        t = data[None, :, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, None:None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, None:None:None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, ::None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        t = data[None, None::None, None]
        tl.static_assert(t.shape == [1, XBLOCK, 1])

        scalar = tl.full([], 1, tl.int32)
        tl.static_assert(scalar.shape == [])

        t = scalar[None]
        tl.static_assert(t.shape == [1])

        t = scalar[None, None]
        tl.static_assert(t.shape == [1, 1])
