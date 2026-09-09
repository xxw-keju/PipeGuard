    @triton.jit
    def temp_in_loop(Z, N: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.full((BLOCK, ), 0, dtype=tl.int32)
        for i in range(N):
            if i == 0:
                temp = tl.full((BLOCK, ), 2, dtype=tl.int32)
                acc = temp
            else:
                acc += tl.full((BLOCK, ), 1, dtype=tl.int32)
            # reuse the temp variable and make sure to check that it isn't creating incorrect IR.
            temp = tl.full((BLOCK, ), 1, dtype=tl.int32)
            acc += temp
        z = Z + tl.arange(0, BLOCK)
        tl.store(z, acc)
