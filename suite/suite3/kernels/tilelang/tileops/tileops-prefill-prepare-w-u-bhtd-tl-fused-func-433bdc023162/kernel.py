# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import math
import tilelang
LOG2E: float = 1.4426950408889634

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32'):
    """Compute Gated DeltaNet prefill ``w`` and ``u`` without global Aw/Au."""
    accum_dtype = 'float32'
    block_C = chunk_size
    num_rounds = int(math.ceil(math.log2(chunk_size))) if chunk_size > 1 else 0

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _fused_func(num_stages, threads=128):

        @T.prim_func
        def prefill_prepare_w_u_bhtd(k: T.Tensor([batch, head, seq_len, dim_k], dtype), v: T.Tensor([batch, head, seq_len, dim_v], dtype), g: T.Tensor([batch, head, seq_len], dtype), beta: T.Tensor([batch, head, seq_len], dtype), w: T.Tensor([batch, head, seq_len, dim_k], dtype), u: T.Tensor([batch, head, seq_len, dim_v], dtype)):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, by):
                k_shared = T.alloc_shared([block_C, dim_k], dtype)
                g_shared = T.alloc_shared([block_C], dtype)
                beta_shared = T.alloc_shared([block_C], dtype)
                k_beta_shared = T.alloc_shared([block_C, dim_k], dtype)
                v_beta_shared = T.alloc_shared([block_C, dim_v], dtype)
                S_shared = T.alloc_shared([block_C, block_C], dtype)
                P_shared = T.alloc_shared([block_C, block_C], dtype)
                g_pos_shared = T.alloc_shared([block_C], accum_dtype)
                g_neg_shared = T.alloc_shared([block_C], accum_dtype)
                gram_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                temp_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                w_frag = T.alloc_fragment([block_C, dim_k], accum_dtype)
                u_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                T.copy(k[bid, hid, by * block_C:(by + 1) * block_C, :], k_shared, disable_tma=True)
                T.copy(g[bid, hid, by * block_C:(by + 1) * block_C], g_shared, disable_tma=True)
                T.copy(beta[bid, hid, by * block_C:(by + 1) * block_C], beta_shared, disable_tma=True)
                for i_s, i_k in T.Parallel(block_C, dim_k):
                    k_beta_shared[i_s, i_k] = k_shared[i_s, i_k] * beta_shared[i_s]
                for i in T.Parallel(block_C):
                    g_pos_shared[i] = T.exp2(g_shared[i] * LOG2E)
                    g_neg_shared[i] = T.exp2(-g_shared[i] * LOG2E)
                T.clear(gram_frag)
                T.gemm(k_beta_shared, k_shared, gram_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    P_shared[i, j] = T.if_then_else(i > j, -gram_frag[i, j] * g_pos_shared[i] * g_neg_shared[j], T.float32(0.0))
                T.clear(S_shared)
                for i in T.Parallel(block_C):
                    S_shared[i, i] = T.float32(1.0)
                for _r in T.Serial(num_rounds):
                    T.clear(temp_frag)
                    T.gemm(P_shared, S_shared, temp_frag)
                    for i, j in T.Parallel(block_C, block_C):
                        S_shared[i, j] = S_shared[i, j] + temp_frag[i, j]
                    T.clear(temp_frag)
                    T.gemm(P_shared, P_shared, temp_frag)
                    T.copy(temp_frag, P_shared)
                T.clear(w_frag)
                T.gemm(S_shared, k_beta_shared, w_frag)
                T.copy(w_frag, w[bid, hid, by * block_C:(by + 1) * block_C, :], disable_tma=True)
                for i, j in T.Parallel(block_C, dim_v):
                    v_beta_shared[i, j] = v[bid, hid, by * block_C + i, j] * beta_shared[i]
                T.clear(u_frag)
                T.gemm(S_shared, v_beta_shared, u_frag)
                T.copy(u_frag, u[bid, hid, by * block_C:(by + 1) * block_C, :], disable_tma=True)
        return prefill_prepare_w_u_bhtd
    return _fused_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
