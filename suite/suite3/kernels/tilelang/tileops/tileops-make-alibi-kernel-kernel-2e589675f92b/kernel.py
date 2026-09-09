# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(seq_len, num_heads, dtype, threads=256, npt=8):
    """Build ALiBi kernel: bias[h, i, j] = -slope_h * |i - j|.

    Slopes: slope_h = 2^(-8*h/H) for head h in [0, H).
    Output shape: (num_heads, seq_len, seq_len).
    Total elements: num_heads * seq_len * seq_len.
    """
    N_total = num_heads * seq_len * seq_len
    S2 = seq_len * seq_len

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
                        exp_val = T.cast(-8.0, 'float32') * T.cast(h + 1, 'float32') / T.cast(num_heads, 'float32')
                        slope = T.exp2(exp_val)
                        dist = T.cast(row - col, 'float32')
                        abs_dist = T.if_then_else(dist > T.cast(0, 'float32'), dist, -dist)
                        out[flat] = T.Cast(dtype, -slope * abs_dist)
        return main
    return kernel

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
