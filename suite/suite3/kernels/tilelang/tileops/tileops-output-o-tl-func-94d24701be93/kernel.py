# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32'):
    """Output projection: (q, k, g, S, v_new) -> o.

    Grid: (num_chunks, batch, head) — fully parallel across chunks.
    Each chunk reads h = S[t] (boundary state at start of chunk) and v_new[t].
    """
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(threads=128):

        @T.prim_func
        def output_o_kernel(q: T.Tensor([batch, head, seq_len, dim_k], dtype), k: T.Tensor([batch, head, seq_len, dim_k], dtype), g: T.Tensor([batch, head, seq_len], dtype), S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype), v_new: T.Tensor([batch, head, seq_len, dim_v], dtype), o: T.Tensor([batch, head, seq_len, dim_v], dtype)):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                T.copy(q[bid, hid, tid * block_C:(tid + 1) * block_C, :], q_c, disable_tma=True)
                T.copy(k[bid, hid, tid * block_C:(tid + 1) * block_C, :], k_c, disable_tma=True)
                T.copy(g[bid, hid, tid * block_C:(tid + 1) * block_C], g_c, disable_tma=True)
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)
                T.copy(v_new[bid, hid, tid * block_C:(tid + 1) * block_C, :], v_new_c, disable_tma=True)
                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * LOG2E)
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * LOG2E), T.float32(0.0))
                T.gemm(attn, v_new_c, o_frag)
                T.copy(o_frag, o[bid, hid, tid * block_C:(tid + 1) * block_C, :], disable_tma=True)
        return output_o_kernel
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
