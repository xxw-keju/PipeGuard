        @triton.jit
        def closure():
            tl.full((16, ), cst, dtype=tl.int32)
