# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang
_dtype_bytes = {'float16': 2, 'bfloat16': 2, 'float32': 4, 'int32': 4}

def _kernel_factory(m: int, n: int, k: int, dtype: str='float16') -> Callable:
    """Bandwidth-bound NT GEMM ``C[m,n] = A[m,k] @ B[n,k]ᵀ`` for small ``m`` (SM90).

    The weight matrix ``B`` is streamed once through a cp.async SMEM ring and
    each B-tile is reused across all ``m`` rows — the regime where arithmetic
    intensity is ``~m`` and saturating HBM, not tensor-core rate, is the goal.
    One ``tvm_thread_allreduce`` over the ``tk`` reduce lanes runs per output
    row.

    Serves both bandwidth-mode kernels of this family: ``SmallBatchGemmKernel``
    at its dispatched ``m``, and ``GemvKernel`` at ``m = 1`` (the matrix-vector
    case is this kernel with a one-row ``A``, not a separate implementation —
    ``for mi in T.serial(1)`` folds away).

    The epilogue write is N-tail guarded, which is load-bearing rather than
    defensive: ``gemv_config`` selects ``block_n = 2`` for ``k >= 12288``, and
    without the guard an odd ``n`` writes one element past ``c[.., n-1]``. The K
    tail needs no mask written here — both tail loads are predicated by the
    buffer extents in the emitted code (see the mainloop comment).

    Args:
        m: Batch rows (``1`` for the GEMV case, else the dispatched small ``m``).
        n: Output columns (weight rows).
        k: Contraction dim; the innermost dim of both ``a`` and ``b``.
        dtype: TileLang dtype string (``"float16"`` / ``"bfloat16"``).

    Returns:
        A JIT factory ``(block_n, reduce_threads, num_stages) -> compiled`` whose
        compiled kernel maps ``(a[m,k], b[n,k]) -> c[m,n]``.
    """
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], compile_flags=['-O3', '-DENABLE_BF16'])
    def _gemm_small_batch_func(block_n: int=1, reduce_threads: int=128, num_stages: int=4) -> Callable:
        tile_k = 128 // (_dtype_bytes[dtype] * 8)
        block_k = reduce_threads * tile_k

        @T.prim_func
        def _gemm_small_batch_main(a: T.Tensor((m, k), dtype), b: T.Tensor((n, k), dtype), c: T.Tensor((m, n), dtype)):
            with T.Kernel(T.ceildiv(n, block_n), threads=(reduce_threads, block_n)) as bn:
                tk = T.get_thread_binding(0)
                tn = T.get_thread_binding(1)
                c_accum = T.alloc_local((m,), accum_dtype)
                T.clear(c_accum)
                b_shared = T.alloc_shared((block_n, block_k), dtype)
                a_local = T.alloc_local((m, tile_k), dtype)
                for bk in T.Pipelined(T.ceildiv(k, block_k), num_stages=num_stages):
                    T.copy(b[bn * block_n, bk * block_k], b_shared, disable_tma=True)
                    for mi in T.serial(m):
                        for _k in T.vectorized(tile_k):
                            a_local[mi, _k] = a[mi, bk * block_k + tk * tile_k + _k]
                    for mi in T.serial(m):
                        for _k in T.serial(tile_k):
                            c_accum[mi] += a_local[mi, _k].astype(accum_dtype) * b_shared[tn, tk * tile_k + _k].astype(accum_dtype)
                c_reduced = T.alloc_local((1,), accum_dtype)
                for mi in T.serial(m):
                    with T.attr(T.comm_reducer(lambda x, y: x + y, [T.Cast(accum_dtype, 0)]), 'reduce_scope', T.reinterpret(T.uint64(0), dtype='handle')):
                        T.evaluate(T.tvm_thread_allreduce(T.uint32(1), c_accum[mi], True, c_reduced[0], tk, dtype='handle'))
                    if bn * block_n + tn < n:
                        c[mi, bn * block_n + tn] = c_reduced[0]
        return _gemm_small_batch_main
    return _gemm_small_batch_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
