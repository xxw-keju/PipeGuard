# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634
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

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32', block_v: int=0, segment_chunks: int=8):
    """Summarize a short reverse chunk segment as X[left] = B + A * X[right]."""
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError('num_chunks must be divisible by segment_chunks')
    num_segments = num_chunks // segment_chunks
    BV = resolve_block_v(dim_v, block_v)
    num_v_tiles = dim_v // BV

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(num_stages, threads=256):

        @T.prim_func
        def dh_segment_summary_kernel(g: T.Tensor([batch, head, seq_len], dtype), dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype), segment_alpha: T.Tensor([batch, head, num_segments], 'float32'), segment_local: T.Tensor([batch, head, num_v_tiles, num_segments, dim_k, BV], dtype)):
            with T.Kernel(num_v_tiles, num_segments, batch * head, threads=threads) as (vid, sid, bhid):
                bid = bhid // head
                hid = bhid - bid * head
                g_c = T.alloc_shared([block_C], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                summary = T.alloc_shared([dim_k, BV], dtype)
                summary_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                v_offset = vid * BV
                alpha_acc = T.alloc_var(T.float32, init=1.0)
                for i, j in T.Parallel(dim_k, BV):
                    summary[i, j] = T.float32(0.0)
                for step in T.Serial(segment_chunks):
                    cid = sid * segment_chunks + (segment_chunks - 1 - step)
                    T.copy(g[bid, hid, cid * block_C:(cid + 1) * block_C], g_c, disable_tma=True)
                    T.copy(dh_local[bid, hid, cid, :, v_offset:v_offset + BV], dh_loc, disable_tma=True)
                    alpha = T.exp2(g_c[block_C - 1] * LOG2E)
                    for i, j in T.Parallel(dim_k, BV):
                        summary_frag[i, j] = dh_loc[i, j] + summary[i, j] * alpha
                    T.copy(summary_frag, summary)
                    alpha_acc = alpha * alpha_acc
                if vid == 0:
                    segment_alpha[bid, hid, sid] = alpha_acc
                T.copy(summary, segment_local[bid, hid, vid, sid, :, :], disable_tma=True)
        return dh_segment_summary_kernel
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
