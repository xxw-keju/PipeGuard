    @tilelang.jit(out_idx=[0])
    def kernel(threads_arg):
        @T.prim_func
        def main(out: T.Tensor((seq_len, d_model), dtype)):
            with T.Kernel(T.ceildiv(half, cols), T.ceildiv(seq_len, rows), threads=threads_arg) as (
                bc,
                br,
            ):
                # Named `divisor`, not `div`: stdlib.h declares a `div` the emitted
                # CUDA would collide with.
                divisor = T.alloc_shared((cols,), "float32")
                angle = T.alloc_fragment((rows, cols), "float32")
                pe = T.alloc_fragment((rows, width), dtype)
                for c in T.Parallel(cols):
                    exponent = (
                        T.cast(bc * cols + c, "float32")
                        * T.cast(2.0, "float32")
                        / T.cast(d_model, "float32")
                    )
                    divisor[c] = T.pow(T.cast(10000.0, "float32"), exponent)
                for r, c in T.Parallel(rows, cols):
                    angle[r, c] = T.cast(br * rows + r, "float32") / divisor[c]
                for r, c in T.Parallel(rows, width):
                    pe[r, c] = T.Cast(
                        dtype,
                        T.if_then_else(
                            c % 2 == 0, T.sin(angle[r, c // 2]), T.cos(angle[r, c // 2])
                        ),
                    )
                T.copy(pe, out[br * rows : (br + 1) * rows, bc * width : (bc + 1) * width])

        return main
