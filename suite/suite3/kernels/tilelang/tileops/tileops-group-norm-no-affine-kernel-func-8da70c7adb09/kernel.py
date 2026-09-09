# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
NARROW_ROW = 2048

def _holds_row_in_registers(D: int, D_padded: int) -> bool:
    """Whether a row of this width is read from global memory into fragments.

    A row whose width the block divides is. A padded one loses the vectorized
    copy to a per-element guard, so it only stays in registers while it is narrow
    enough that staging through shared memory would cost more.

    Args:
        D: Row length.
        D_padded: *D* rounded up to a width the block divides.
    """
    return D_padded == D or D_padded <= NARROW_ROW

def make_row_reduce(block_m, n, n_padded, eps):
    """Create the macro reducing a loaded fp32 row block to mean and rstd.

    Consumes ``x_f32`` and overwrites it with the centered squares. The load
    stays at the call sites, which read the row block in the dtype the tensor
    holds and keep it for the output pass.

    Args:
        block_m: Rows per block.
        n: Row length.
        n_padded: *n* rounded up to :data:`ALIGNMENT`.
        eps: Epsilon for numerical stability.

    Returns:
        A ``@T.macro`` taking ``(x_f32, acc, mean_val, rstd)``.
    """
    pad_count = n_padded - n

    @T.macro
    def row_reduce(x_f32, acc, mean_val, rstd):
        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            mean_val[i] = acc[i] / float(n)
        for i, j in T.Parallel(block_m, n_padded):
            x_f32[i, j] = (x_f32[i, j] - mean_val[i]) * (x_f32[i, j] - mean_val[i])
        T.reduce_sum(x_f32, acc, dim=1)
        for i in T.Parallel(block_m):
            rstd[i] = T.rsqrt((acc[i] - float(pad_count) * mean_val[i] * mean_val[i]) / float(n) + eps)
    return row_reduce

def make_shifted_row_reduce(block_m, n, eps):
    """Create the macro turning an already-shifted row block into mean and rstd.

    Reads the two fragments the load fills and rewrites neither: ``d`` holds
    ``x - shift`` and ``sq`` holds its square. ``mean_val`` comes back shifted by
    the same amount, so the output pass reads ``d[i, j] - mean_val[i]`` and gets
    ``x - E[x]``.

    Two things the caller supplies:

    - ``shift`` is a value from the row itself. Without it the difference of the
      two sums loses the variance to cancellation as soon as a row's mean
      outgrows its spread.
    - A pad column holds zero in both fragments, which is why no correction term
      for the padding appears here.

    Args:
        block_m: Rows per block.
        n: Row length.
        eps: Epsilon for numerical stability.

    Returns:
        A ``@T.macro`` taking ``(d, sq, acc_shifted, acc_squares, mean_val, rstd)``.
    """

    @T.macro
    def shifted_row_reduce(d, sq, acc_shifted, acc_squares, mean_val, rstd):
        T.reduce_sum(d, acc_shifted, dim=1)
        T.reduce_sum(sq, acc_squares, dim=1)
        for i in T.Parallel(block_m):
            mean_val[i] = acc_shifted[i] / float(n)
            rstd[i] = T.rsqrt(acc_squares[i] / float(n) - mean_val[i] * mean_val[i] + eps)
    return shifted_row_reduce
ALIGNMENT = 256
_ROW_SMEM_BUDGET_BYTES = 48 * 1024

def row_padding(n: int, elem_bytes: int) -> int:
    """Row width to pad *n* to: the next power of two while it fits in shared.

    Only a power of two is divided by every entry of :data:`_CANDIDATE_THREADS`;
    the 256-element alignment leaves an odd factor that caps a block at eight
    warps, which costs 3% on a row of 7168. Never below :data:`ALIGNMENT`, which
    the 128-thread default needs.
    """
    pow2 = 1 << (n - 1).bit_length()
    if pow2 * elem_bytes <= _ROW_SMEM_BUDGET_BYTES:
        return max(pow2, ALIGNMENT)
    return -(-n // ALIGNMENT) * ALIGNMENT

def _kernel_factory(M, D, eps, dtype):
    """Build a row-wise normalization kernel for shape (M, D) without affine.

    Same numerics and same boundary handling as `_group_norm_kernel`,
    but omits the trailing weight/bias multiply-add — output is
    ``(x - mean) * rstd``. Used for the no-affine variants of GroupNorm and
    InstanceNorm.

    Args:
        M: Number of rows = N * G.
        D: Row length = (C / G) * spatial_size.
        eps: Epsilon for numerical stability.
        dtype: TileLang dtype string.
    """
    D_padded = row_padding(D, 4 if dtype == 'float32' else 2)

    @tilelang.jit(out_idx=[1])
    def _func(block_m, threads):
        masked = D_padded != D
        register_direct = _holds_row_in_registers(D, D_padded)
        guarded = masked or M % block_m != 0
        if register_direct:
            row_reduce = make_shifted_row_reduce(block_m, D, eps)
        else:
            row_reduce = make_row_reduce(block_m, D, D_padded, eps)

        @T.prim_func
        def main(x: T.Tensor[(M, D), dtype], y: T.Tensor[(M, D), dtype]):
            with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
                if register_direct:
                    centered_row = T.alloc_fragment((block_m, D_padded), 'float32')
                    squares = T.alloc_fragment((block_m, D_padded), 'float32')
                    shift = T.alloc_fragment((block_m,), 'float32')
                    acc_squares = T.alloc_fragment((block_m,), 'float32')
                else:
                    shared_buf = T.alloc_shared((block_m, D_padded), dtype)
                    x_f32 = T.alloc_fragment((block_m, D_padded), 'float32')
                acc = T.alloc_fragment((block_m,), 'float32')
                mean_val = T.alloc_fragment((block_m,), 'float32')
                rstd = T.alloc_fragment((block_m,), 'float32')
                if register_direct:
                    for i in T.Parallel(block_m):
                        shift[i] = T.cast(x[T.min(pid_m * block_m + i, M - 1), 0], 'float32')
                    if guarded:
                        for i, j in T.Parallel(block_m, D_padded):
                            v = T.if_then_else(T.And(pid_m * block_m + i < M, j < D), T.cast(x[pid_m * block_m + i, j], 'float32') - shift[i], T.cast(0.0, 'float32'))
                            centered_row[i, j] = v
                            squares[i, j] = v * v
                    else:
                        for i, j in T.Parallel(block_m, D_padded):
                            v = T.cast(x[pid_m * block_m + i, j], 'float32') - shift[i]
                            centered_row[i, j] = v
                            squares[i, j] = v * v
                    row_reduce(centered_row, squares, acc, acc_squares, mean_val, rstd)
                else:
                    for i, j in T.Parallel(block_m, D_padded):
                        shared_buf[i, j] = T.if_then_else(T.And(pid_m * block_m + i < M, j < D), x[pid_m * block_m + i, j], T.cast(0.0, dtype))
                        x_f32[i, j] = T.cast(shared_buf[i, j], 'float32')
                    row_reduce(x_f32, acc, mean_val, rstd)
                if guarded:
                    for i, j in T.Parallel(block_m, D_padded):
                        if T.And(pid_m * block_m + i < M, j < D):
                            y[pid_m * block_m + i, j] = T.cast((T.cast(centered_row[i, j] if register_direct else shared_buf[i, j], 'float32') - mean_val[i]) * rstd[i], dtype)
                else:
                    for i, j in T.Parallel(block_m, D_padded):
                        y[pid_m * block_m + i, j] = T.cast((centered_row[i, j] - mean_val[i]) * rstd[i], dtype)
        return main
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
