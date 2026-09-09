    @tilelang.jit(out_idx=[2])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(
            x: T.Tensor((N,), dtype),
            weight: T.Tensor((C,), dtype),
            y: T.Tensor((N,), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    k = i * npt_arg + j
                    idx = bx * block_size + k
                    val = x_reg[k]
                    ch = (idx // inner_size) % C
                    w = weight[ch]
                    zero = T.cast(0, val.dtype)
                    y_reg[k] = T.if_then_else(val > zero, val, w * val)
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main
