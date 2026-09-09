# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634

def _make_dh_correction_from_carry_macro(block_C: int, dim_k: int, BV: int, accum_dtype: str):
    """Create the shared chunk correction macro for an incoming successor carry."""

    @T.macro
    def correction_from_carry(g_c, k_c, v_new_c, h_c, dh_buf, k_scaled, dP, dg_c, du_corr_c, dh_h_tmp, du_corr_frag, dP_frag, d_g_pos, d_g_last_partial, d_g_last_scalar1, d_g_last_scalar2, dk_corr_partial, du_corr, dg_corr_partial, dw_corr_partial, bid, hid, vid, chunk_offset, v_offset):
        for pn, sk in T.Parallel(block_C, dim_k):
            k_scaled[pn, sk] = k_c[pn, sk] * T.exp2((g_c[block_C - 1] - g_c[pn]) * LOG2E)
        T.clear(du_corr_frag)
        T.gemm(k_scaled, dh_buf, du_corr_frag)
        T.copy(du_corr_frag, du_corr_c)
        T.copy(du_corr_c, du_corr[bid, hid, chunk_offset:chunk_offset + block_C, v_offset:v_offset + BV], disable_tma=True)
        T.clear(dP_frag)
        T.gemm(du_corr_c, h_c, dP_frag, transpose_B=True)
        for n, kk in T.Parallel(block_C, dim_k):
            dw_corr_partial[bid, hid, vid, chunk_offset + n, kk] = -dP_frag[n, kk] * T.exp2((g_c[n] + g_c[block_C - 1]) * LOG2E)
        T.clear(dP_frag)
        T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
        T.copy(dP_frag, dP)
        for n, kk in T.Parallel(block_C, dim_k):
            dk_corr_partial[bid, hid, vid, chunk_offset + n, kk] = dP[n, kk] * T.exp2((g_c[block_C - 1] - g_c[n]) * LOG2E)
        for n, kk in T.Parallel(block_C, dim_k):
            dP[n, kk] = dP[n, kk] * k_scaled[n, kk]
        T.reduce_sum(dP, d_g_pos, dim=1)
        for n in T.Parallel(block_C):
            dg_c[n] = -d_g_pos[n]
        for i, j in T.Parallel(dim_k, BV):
            dh_h_tmp[i, j] = dh_buf[i, j] * h_c[i, j]
        T.reduce_sum(dh_h_tmp, d_g_last_partial, dim=1)
        T.reduce_sum(d_g_last_partial, d_g_last_scalar1, dim=0)
        T.reduce_sum(d_g_pos, d_g_last_scalar2, dim=0)
        dg_c[block_C - 1] = dg_c[block_C - 1] + d_g_last_scalar1[0] * T.exp2(g_c[block_C - 1] * LOG2E) + d_g_last_scalar2[0]
        for i in T.Parallel(block_C):
            dg_corr_partial[bid, hid, vid, chunk_offset + i] = dg_c[i]
    return correction_from_carry
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
    """Sequential backward dh recurrence with corrections.

    Grid: (num_v_tiles, batch, head) — sequential over chunks (backward),
    parallel over V tiles.
    Reads dh_local from bwd_parallel, propagates dh backward, and computes
    corrections for dk, du, dg that depend on dh_buf from other chunks.

    Outputs: dk_correction_partial, du_correction, dg_correction_partial,
    dw_correction_partial. The partial dk/dg/dw outputs must be reduced over
    the V-tile dimension by the caller. This keeps the per-CTA state buffer
    small enough for d128.
    """
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = resolve_block_v(dim_v, block_v)
    num_v_tiles = dim_v // BV

    @tilelang.jit(out_idx=[-4, -3, -2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(num_stages, threads=256):
        correction_from_carry = _make_dh_correction_from_carry_macro(block_C, dim_k, BV, accum_dtype)

        @T.prim_func
        def dh_recurrence_bwd_kernel(g: T.Tensor([batch, head, seq_len], dtype), k: T.Tensor([batch, head, seq_len, dim_k], dtype), v_new: T.Tensor([batch, head, seq_len, dim_v], dtype), S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype), dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], dtype), dk_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype), du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype), dg_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len], dtype), dw_corr_partial: T.Tensor([batch, head, num_v_tiles, seq_len, dim_k], dtype)):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                g_c = T.alloc_shared([block_C], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                dh_loc = T.alloc_shared([dim_k, BV], dtype)
                k_scaled = T.alloc_shared([block_C, dim_k], dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                dg_c = T.alloc_shared([block_C], dtype)
                du_corr_c = T.alloc_shared([block_C, BV], dtype)
                dh_buf = T.alloc_shared([dim_k, BV], dtype)
                dh_h_tmp = T.alloc_shared([dim_k, BV], dtype)
                d_g_pos = T.alloc_shared([block_C], dtype)
                d_g_last_partial = T.alloc_shared([dim_k], dtype)
                d_g_last_scalar1 = T.alloc_shared([1], accum_dtype)
                d_g_last_scalar2 = T.alloc_shared([1], accum_dtype)
                dh_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                v_offset = vid * BV
                for i, j in T.Parallel(dim_k, BV):
                    dh_buf[i, j] = T.float32(0.0)
                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(g[bid, hid, t_bwd * block_C:(t_bwd + 1) * block_C], g_c, disable_tma=True)
                    T.copy(k[bid, hid, t_bwd * block_C:(t_bwd + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(v_new[bid, hid, t_bwd * block_C:(t_bwd + 1) * block_C, v_offset:v_offset + BV], v_new_c, disable_tma=True)
                    T.copy(S[bid, hid, t_bwd, :, v_offset:v_offset + BV], h_c, disable_tma=True)
                    T.copy(dh_local[bid, hid, t_bwd, :, v_offset:v_offset + BV], dh_loc, disable_tma=True)
                    for i, j in T.Parallel(dim_k, BV):
                        dh_frag[i, j] = dh_loc[i, j] + dh_buf[i, j] * T.exp2(g_c[block_C - 1] * LOG2E)
                    correction_from_carry(g_c, k_c, v_new_c, h_c, dh_buf, k_scaled, dP, dg_c, du_corr_c, dh_h_tmp, du_corr_frag, dP_frag, d_g_pos, d_g_last_partial, d_g_last_scalar1, d_g_last_scalar2, dk_corr_partial, du_corr, dg_corr_partial, dw_corr_partial, bid, hid, vid, t_bwd * block_C, v_offset)
                    T.copy(dh_frag, dh_buf)
        return dh_recurrence_bwd_kernel
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
