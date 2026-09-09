# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
ALIGNMENT = 256

def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*, which must be positive."""
    if alignment <= 0:
        raise ValueError(f'alignment must be positive, got {alignment}')
    return (n + alignment - 1) // alignment * alignment

def _kernel_factory(M, N, eps, dtype):
    N_padded = align_up(N, ALIGNMENT)
    pad_count = N_padded - N

    @tilelang.jit(out_idx=[4, 5])
    def _func(block_m, threads):
        row_guard = M % block_m != 0

        @T.prim_func
        def main(x: T.Tensor[(M, N_padded), dtype], residual: T.Tensor[(M, N_padded), dtype], weight: T.Tensor[(N_padded,), dtype], bias: T.Tensor[(N_padded,), dtype], y: T.Tensor[(M, N_padded), dtype], residual_out: T.Tensor[(M, N_padded), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                r_local = T.alloc_fragment((block_m, N_padded), dtype)
                add_f32 = T.alloc_fragment((block_m, N_padded), 'float32')
                acc = T.alloc_fragment((block_m,), 'float32')
                mean_val = T.alloc_fragment((block_m,), 'float32')
                rstd = T.alloc_fragment((block_m,), 'float32')
                if row_guard:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = T.if_then_else(pid_m * block_m + i < M, x[pid_m * block_m + i, j], T.cast(0.0, dtype))
                    for i, j in T.Parallel(block_m, N_padded):
                        r_local[i, j] = T.if_then_else(pid_m * block_m + i < M, residual[pid_m * block_m + i, j], T.cast(0.0, dtype))
                else:
                    for i, j in T.Parallel(block_m, N_padded):
                        x_local[i, j] = x[pid_m * block_m + i, j]
                    for i, j in T.Parallel(block_m, N_padded):
                        r_local[i, j] = residual[pid_m * block_m + i, j]
                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = T.cast(x_local[i, j], 'float32') + T.cast(r_local[i, j], 'float32')
                for i, j in T.Parallel(block_m, N_padded):
                    x_local[i, j] = add_f32[i, j]
                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    mean_val[i] = acc[i] / float(N)
                for i, j in T.Parallel(block_m, N_padded):
                    add_f32[i, j] = (add_f32[i, j] - mean_val[i]) * (add_f32[i, j] - mean_val[i])
                T.reduce_sum(add_f32, acc, dim=1)
                for i in T.Parallel(block_m):
                    rstd[i] = T.rsqrt((acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(N) + eps)
                for i, j in T.Parallel(block_m, N_padded):
                    if not row_guard or pid_m * block_m + i < M:
                        y[pid_m * block_m + i, j] = T.cast((T.cast(x_local[i, j], 'float32') - mean_val[i]) * rstd[i] * T.cast(weight[j], 'float32') + T.cast(bias[j], 'float32'), dtype)
                for i, j in T.Parallel(block_m, N_padded):
                    if not row_guard or pid_m * block_m + i < M:
                        residual_out[pid_m * block_m + i, j] = x_local[i, j]
        return main
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
