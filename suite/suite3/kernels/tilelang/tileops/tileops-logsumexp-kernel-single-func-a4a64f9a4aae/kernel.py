# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
ALIGNMENT = 256
DEFAULT_ALIGNMENT: int = ALIGNMENT

def align_up(n: int, alignment: int) -> int:
    """Round *n* up to the nearest multiple of *alignment*, which must be positive."""
    if alignment <= 0:
        raise ValueError(f'alignment must be positive, got {alignment}')
    return (n + alignment - 1) // alignment * alignment

def _kernel_factory(M: int, N: int, dtype: str):
    """Build a single-tile logsumexp kernel (N fits in smem).

    Accepts an ``(M, N)`` input tensor.  When ``N`` is not a multiple of
    ``DEFAULT_ALIGNMENT``, the kernel uses element-wise ``T.if_then_else``
    loads that substitute ``-inf`` for out-of-bounds columns (kernel-side
    boundary handling).  When ``N`` is already aligned, the fast ``T.copy``
    path is used.
    """
    N_padded = align_up(N, DEFAULT_ALIGNMENT)
    _needs_pad = N_padded != N
    _neg_inf = float('-inf')

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):

        @T.prim_func
        def main(x: T.Tensor[(M, N), dtype], y: T.Tensor[(M,), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                shared_buf = T.alloc_shared((block_m, N_padded), dtype)
                x_local = T.alloc_fragment((block_m, N_padded), dtype)
                x_f32 = T.alloc_fragment((block_m, N_padded), 'float32')
                row_max = T.alloc_fragment((block_m,), 'float32')
                row_sum = T.alloc_fragment((block_m,), 'float32')
                if _needs_pad:
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, j < N), T.cast(x[pid_m * block_m + i, j], 'float32'), T.cast(_neg_inf, 'float32'))
                else:
                    T.copy(x[pid_m * block_m, 0], shared_buf)
                    T.copy(shared_buf, x_local)
                    for i in T.serial(block_m):
                        for j in T.Parallel(N_padded):
                            x_f32[i, j] = T.cast(x_local[i, j], 'float32')
                T.fill(row_max, -T.infinity('float32'))
                T.reduce_max(x_f32, row_max, dim=1, clear=False)
                for i in T.serial(block_m):
                    for j in T.Parallel(N_padded):
                        x_f32[i, j] = T.exp(x_f32[i, j] - row_max[i])
                T.reduce_sum(x_f32, row_sum, dim=1)
                out_local = T.alloc_fragment((block_m,), dtype)
                for i in T.Parallel(block_m):
                    out_local[i] = row_max[i] + T.log(row_sum[i])
                T.copy(out_local, y[pid_m * block_m])
        return main
    return _func

def build_kernel(case):
    """Construct the selected TileLang device entry without launching it."""
    return _kernel_factory(**case["build"]["factory_arguments"])(**case["build"]["kernel_arguments"])
