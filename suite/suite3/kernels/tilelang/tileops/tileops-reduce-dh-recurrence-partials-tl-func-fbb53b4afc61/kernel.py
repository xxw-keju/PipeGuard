# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
GEMM_MIN_N = 16

def resolve_block_v(dim_v: int, block_v: int) -> int:
    """Return the effective V-tile width; ``block_v <= 0`` means no tiling.

    Raises:
        ValueError: if the resolved width is below ``GEMM_MIN_N`` or does not
            divide ``dim_v``.
    """
    bv = dim_v if block_v <= 0 else block_v
    if bv < GEMM_MIN_N:
        raise ValueError(f'V-tile width {bv} (dim_v={dim_v}, block_v={block_v}) is below the minimum T.gemm N extent ({GEMM_MIN_N}); use block_v >= {GEMM_MIN_N}, or 0 for no tiling with dim_v >= {GEMM_MIN_N}')
    if dim_v % bv != 0:
        raise ValueError(f'dim_v ({dim_v}) must be divisible by block_v ({bv})')
    return bv

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32', block_v: int=0):
    """Reduce V-tiled dh-recurrence correction partials.

    ``dk``, ``dg``, and ``dw`` corrections are additive across V tiles.
    Keeping this reduction inside TileLang avoids generic torch reductions in
    the full backward hot path.
    """
    block_C = chunk_size
    BV = resolve_block_v(dim_v, block_v)
    num_v_tiles = dim_v // BV

    @tilelang.jit(out_idx=[-3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(threads=256):

        @T.prim_func
        def reduce_dh_recurrence_partials(dk_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype), dg_partial: T.Tensor([batch, head, num_v_tiles, seq_len], dtype), dw_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype), dk: T.Tensor([batch, head, seq_len, dim_k], dtype), dg: T.Tensor([batch, head, seq_len], dtype), dw: T.Tensor([batch, head, seq_len, dim_k], dtype)):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, cid):
                for n, kk in T.Parallel(block_C, dim_k):
                    dk_acc = T.alloc_var(T.float32, init=0.0)
                    dw_acc = T.alloc_var(T.float32, init=0.0)
                    for vid in T.Serial(num_v_tiles):
                        dk_acc += T.cast(dk_partial[bid, hid, vid, cid * block_C + n, kk], 'float32')
                        dw_acc += T.cast(dw_partial[bid, hid, vid, cid * block_C + n, kk], 'float32')
                    dk[bid, hid, cid * block_C + n, kk] = T.cast(dk_acc, dtype)
                    dw[bid, hid, cid * block_C + n, kk] = T.cast(dw_acc, dtype)
                for n in T.Parallel(block_C):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for vid in T.Serial(num_v_tiles):
                        acc += T.cast(dg_partial[bid, hid, vid, cid * block_C + n], 'float32')
                    dg[bid, hid, cid * block_C + n] = T.cast(acc, dtype)
        return reduce_dh_recurrence_partials
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
