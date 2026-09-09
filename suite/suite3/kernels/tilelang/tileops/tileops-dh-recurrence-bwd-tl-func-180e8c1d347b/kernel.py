# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32'):
    """Sequential backward dh recurrence with corrections.

    Grid: (batch, head) -- sequential over chunks (backward).
    Reads dh_local from bwd_parallel, propagates dh backward, and computes
    corrections for dk, du that depend on dh_buf from other chunks.

    The correct recurrence for ungated DeltaNet is:
        dh_c = dh_local[c] + (I - W_c^T @ K_c) @ dh_{c+1}
    because h_{c+1} = h_c + k^T @ v_new and v_new = u - w @ h_c,
    so dh_{c+1}/dh_c = I - k^T @ w (not I).

    dw_corr is computed outside this kernel as a batched matmul for efficiency.

    Outputs: dk_correction, du_correction
    """
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(num_stages, threads=256):

        @T.prim_func
        def dh_recurrence_bwd_kernel(k: T.Tensor([batch, head, seq_len, dim_k], dtype), w: T.Tensor([batch, head, seq_len, dim_k], dtype), v_new: T.Tensor([batch, head, seq_len, dim_v], dtype), dh_local: T.Tensor([batch, head, num_chunks, dim_k, dim_v], accum_dtype), dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype), du_corr: T.Tensor([batch, head, seq_len, dim_v], dtype)):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                w_c = T.alloc_shared([block_C, dim_k], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                dh_loc = T.alloc_shared([dim_k, dim_v], accum_dtype)
                dP = T.alloc_shared([block_C, dim_k], dtype)
                dh_buf = T.alloc_shared([dim_k, dim_v], dtype)
                dh_fp32 = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                du_corr_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                dP_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                wk_dh_frag = T.alloc_fragment([dim_k, dim_v], accum_dtype)
                k_dh_shared = T.alloc_shared([block_C, dim_v], dtype)
                for i, j in T.Parallel(dim_k, dim_v):
                    dh_fp32[i, j] = T.float32(0.0)
                for t in T.Pipelined(num_chunks, num_stages=num_stages):
                    t_bwd = num_chunks - 1 - t
                    T.copy(k[bid, hid, t_bwd * block_C:(t_bwd + 1) * block_C, :], k_c, disable_tma=True)
                    T.copy(w[bid, hid, t_bwd * block_C:(t_bwd + 1) * block_C, :], w_c, disable_tma=True)
                    T.copy(v_new[bid, hid, t_bwd * block_C:(t_bwd + 1) * block_C, :], v_new_c, disable_tma=True)
                    T.copy(dh_local[bid, hid, t_bwd, :, :], dh_loc, disable_tma=True)
                    T.copy(dh_fp32, dh_buf)
                    T.clear(du_corr_frag)
                    T.gemm(k_c, dh_buf, du_corr_frag)
                    T.copy(du_corr_frag, du_corr[bid, hid, t_bwd * block_C:(t_bwd + 1) * block_C, :], disable_tma=True)
                    T.copy(du_corr_frag, k_dh_shared)
                    T.clear(dP_frag)
                    T.gemm(v_new_c, dh_buf, dP_frag, transpose_B=True)
                    T.copy(dP_frag, dP)
                    for n, kk in T.Parallel(block_C, dim_k):
                        dk_corr[bid, hid, t_bwd * block_C + n, kk] = dP[n, kk]
                    T.clear(wk_dh_frag)
                    T.gemm(w_c, k_dh_shared, wk_dh_frag, transpose_A=True)
                    for i, j in T.Parallel(dim_k, dim_v):
                        dh_fp32[i, j] = dh_fp32[i, j] + dh_loc[i, j] - wk_dh_frag[i, j]
        return dh_recurrence_bwd_kernel
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
