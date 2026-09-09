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
    """State recurrence: (k, w, u, S_0) -> (S, v_new).

    Grid: (num_v_tiles, batch, head) -- sequential over chunks, parallel over V tiles.
    Outputs per-chunk boundary states S and intermediate v_new for output_o.

    Args:
        block_v: V-tile size. 0 means no tiling (block_v = dim_v).
    """
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C
    BV = resolve_block_v(dim_v, block_v)
    num_v_tiles = dim_v // BV

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(num_stages, threads=128):

        @T.prim_func
        def h_recurrence_kernel(k: T.Tensor([batch, head, seq_len, dim_k], dtype), w: T.Tensor([batch, head, seq_len, dim_k], dtype), u: T.Tensor([batch, head, seq_len, dim_v], dtype), S_0: T.Tensor([batch, head, dim_k, dim_v], dtype), S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], accum_dtype), v_new: T.Tensor([batch, head, seq_len, dim_v], dtype)):
            with T.Kernel(num_v_tiles, batch, head, threads=threads) as (vid, bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                u_c = T.alloc_shared([block_C, BV], dtype)
                h_c = T.alloc_shared([dim_k, BV], dtype)
                v_new_c = T.alloc_shared([block_C, BV], dtype)
                ws_frag = T.alloc_fragment([block_C, BV], accum_dtype)
                h_fp32 = T.alloc_fragment([dim_k, BV], accum_dtype)
                v_offset = vid * BV
                T.copy(S_0[bid, hid, :, v_offset:v_offset + BV], h_c, disable_tma=True)
                for i, j in T.Parallel(dim_k, BV):
                    h_fp32[i, j] = T.cast(h_c[i, j], accum_dtype)
                for i, j in T.Parallel(dim_k, BV):
                    S[bid, hid, 0, i, v_offset + j] = h_fp32[i, j]
                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(k[bid, hid, t * block_C:(t + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(w[bid, hid, t * block_C:(t + 1) * block_C, :], w_c, disable_tma=True)
                    T.copy(u[bid, hid, t * block_C:(t + 1) * block_C, v_offset:v_offset + BV], u_c, disable_tma=True)
                    T.copy(h_fp32, h_c)
                    T.clear(ws_frag)
                    T.gemm(w_c, h_c, ws_frag)
                    for i, j in T.Parallel(block_C, BV):
                        v_new_c[i, j] = u_c[i, j] - ws_frag[i, j]
                    T.copy(v_new_c, v_new[bid, hid, t * block_C:(t + 1) * block_C, v_offset:v_offset + BV], disable_tma=True)
                    T.gemm(k_c, v_new_c, h_fp32, transpose_A=True)
                    for i, j in T.Parallel(dim_k, BV):
                        S[bid, hid, t + 1, i, v_offset + j] = h_fp32[i, j]
        return h_recurrence_kernel
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
