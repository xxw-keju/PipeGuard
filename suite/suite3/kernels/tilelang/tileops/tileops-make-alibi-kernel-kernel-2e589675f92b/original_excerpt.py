    @tilelang.jit(out_idx=[0])
    def kernel(threads_arg, npt_arg):
        block_size = threads_arg * npt_arg

        @T.prim_func
        def main(out: T.Tensor((N_total,), dtype)):
            with T.Kernel(T.ceildiv(N_total, block_size), threads=threads_arg) as bx:
                for i, j in T.Parallel(threads_arg, npt_arg):
                    flat = (bx * threads_arg + i) * npt_arg + j
                    if flat < N_total:
                        h = flat // S2
                        rem = flat % S2
                        row = rem // seq_len
                        col = rem % seq_len
                        # slope = 2^(-8 * (h+1) / num_heads)
                        exp_val = (
                            T.cast(-8.0, "float32")
                            * T.cast(h + 1, "float32")
                            / T.cast(num_heads, "float32")
                        )
                        slope = T.exp2(exp_val)
                        dist = T.cast(row - col, "float32")
                        # Use abs via if_then_else since T.abs may not handle int
                        abs_dist = T.if_then_else(dist > T.cast(0, "float32"), dist, -dist)
                        out[flat] = T.Cast(dtype, -slope * abs_dist)

        return main
