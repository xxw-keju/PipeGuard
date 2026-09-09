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

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32', block_v: int=0, segment_chunks: int=8):
    """Reverse scan segment summaries to produce each segment's successor carry."""
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C
    if num_chunks % segment_chunks != 0:
        raise ValueError('num_chunks must be divisible by segment_chunks')
    num_segments = num_chunks // segment_chunks
    BV = resolve_block_v(dim_v, block_v)
    num_v_tiles = dim_v // BV

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(num_stages, threads=256):

        @T.prim_func
        def dh_segment_boundary_scan_kernel(segment_alpha: T.Tensor([batch, head, num_segments], 'float32'), segment_local: T.Tensor([batch, head, num_v_tiles, num_segments, dim_k, BV], dtype), segment_carry_after: T.Tensor([batch, head, num_v_tiles, num_segments, dim_k, BV], dtype)):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                local = T.alloc_shared([dim_k, BV], dtype)
                carry = T.alloc_shared([dim_k, BV], dtype)
                carry_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                for i, j in T.Parallel(dim_k, BV):
                    carry[i, j] = T.float32(0.0)
                for step in T.Pipelined(num_segments, num_stages=num_stages):
                    sid = num_segments - 1 - step
                    T.copy(carry, segment_carry_after[bid, hid, vid, sid, :, :], disable_tma=True)
                    T.copy(segment_local[bid, hid, vid, sid, :, :], local, disable_tma=True)
                    alpha = segment_alpha[bid, hid, sid]
                    for i, j in T.Parallel(dim_k, BV):
                        carry_frag[i, j] = local[i, j] + carry[i, j] * alpha
                    T.copy(carry_frag, carry)
        return dh_segment_boundary_scan_kernel
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
