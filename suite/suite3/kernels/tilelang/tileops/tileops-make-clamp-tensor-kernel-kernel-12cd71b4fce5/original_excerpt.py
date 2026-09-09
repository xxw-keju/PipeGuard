    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            bound: T.Tensor((N,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                bound_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                T.copy(bound[bx * block_size : (bx + 1) * block_size], bound_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    x32 = T.cast(x_reg[k], "float32")
                    b32 = T.cast(bound_reg[k], "float32")
                    r = T.max(x32, b32) if take_max else T.min(x32, b32)
                    # fmaxf / fminf return their non-NaN operand; torch returns NaN.
                    r = T.if_then_else(T.isnan(b32), b32, r)
                    r = T.if_then_else(T.isnan(x32), x32, r)
                    x_reg[k] = T.Cast(dtype, r)
                T.copy(x_reg, y[bx * block_size : (bx + 1) * block_size])

        return main
