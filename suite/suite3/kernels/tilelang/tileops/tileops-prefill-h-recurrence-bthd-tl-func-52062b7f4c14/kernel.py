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

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32', block_v: int=0):
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = resolve_block_v(dim_v, block_v)
    num_v_tiles = dim_v // BV

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(num_stages, threads=128):

        @T.prim_func
        def h_recurrence_bthd(k: T.Tensor([batch, seq_len, head, dim_k], dtype), g: T.Tensor([batch, seq_len, head], dtype), w: T.Tensor([batch, seq_len, head, dim_k], dtype), u: T.Tensor([batch, seq_len, head, dim_v], dtype), S_0: T.Tensor([batch, head, dim_k, dim_v], dtype), S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype), v_new: T.Tensor([batch, seq_len, head, dim_v], dtype)):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_next_frag = T.alloc_fragment([dim_k, BV], accum_dtype)
                v_offset = vid * BV
                T.copy(S_0[bid, hid, :, v_offset:v_offset + BV], h_c, disable_tma=True)
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_c[i, j]
                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    base = t * block_C
                    for i, d in T.Parallel(block_C, dim_k):
                        k_c[i, d] = k[bid, base + i, hid, d]
                        w_c[i, d] = w[bid, base + i, hid, d]
                    for i in T.Parallel(block_C):
                        g_c[i] = g[bid, base + i, hid]
                    for i, d in T.Parallel(block_C, BV):
                        u_c[i, d] = u[bid, base + i, hid, v_offset + d]
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j] * T.exp2((g_c[i] + g_c[block_C - 1]) * LOG2E)
                    for i, j in T.Parallel(block_C, BV):
                        v_new[bid, base + i, hid, v_offset + j] = v_new_c[i, j]
                    for n, j in T.Parallel(block_C, BV):
                        v_new_c[n, j] = v_new_c[n, j] * T.exp2((g_c[block_C - 1] - g_c[n]) * LOG2E)
                    for i, j in T.Parallel(dim_k, BV):
                        h_next_frag[i, j] = h_c[i, j] * T.exp2(g_c[block_C - 1] * LOG2E)
                    T.gemm(k_c, v_new_c, h_next_frag, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(h_next_frag, h_c)
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, t + 1, i, v_offset + j] = h_c[i, j]
        return h_recurrence_bthd
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
