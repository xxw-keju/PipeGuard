    @tilelang.jit(out_idx=[1])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(x: T.Tensor((N,), dtype), y: T.Tensor((N,), dtype)):
            with T.Kernel(T.ceildiv(N, block_size), threads=threads_arg) as bx:
                x_reg = T.alloc_fragment((block_size,), dtype)
                y_reg = T.alloc_fragment((block_size,), dtype)
                T.copy(x[bx * block_size : (bx + 1) * block_size], x_reg)
                for i, j in T.Parallel(threads_arg, npt_arg):
                    val = x_reg[i * npt_arg + j]
                    v32 = T.cast(val, "float32")
                    b = T.cast(beta, "float32")
                    t = T.cast(threshold, "float32")
                    one = T.cast(1.0, "float32")
                    scaled = v32 * b
                    softplus = log_for_output_precision(val, one + T.exp(scaled)) / b
                    y_reg[i * npt_arg + j] = T.if_then_else(
                        scaled > t,
                        val,
                        T.Cast(val.dtype, softplus),
                    )
                T.copy(y_reg, y[bx * block_size : (bx + 1) * block_size])

        return main
