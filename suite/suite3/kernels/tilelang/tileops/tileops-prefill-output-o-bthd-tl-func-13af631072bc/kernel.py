# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dim_v: int, dtype: str='float32'):
    accum_dtype = 'float32'
    block_C = chunk_size
    num_chunks = seq_len // block_C

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(threads=128):

        @T.prim_func
        def output_o_bthd(q: T.Tensor([batch, seq_len, head, dim_k], dtype), k: T.Tensor([batch, seq_len, head, dim_k], dtype), g: T.Tensor([batch, seq_len, head], dtype), S: T.Tensor([batch, head, num_chunks + 1, dim_k, dim_v], dtype), v_new: T.Tensor([batch, seq_len, head, dim_v], dtype), o: T.Tensor([batch, seq_len, head, dim_v], dtype)):
            with T.Kernel(num_chunks, batch, head, threads=threads) as (tid, bid, hid):
                base = tid * block_C
                q_c = T.alloc_shared([block_C, dim_k], dtype)
                k_c = T.alloc_shared([block_C, dim_k], dtype)
                g_c = T.alloc_shared([block_C], dtype)
                h_c = T.alloc_shared([dim_k, dim_v], dtype)
                v_new_c = T.alloc_shared([block_C, dim_v], dtype)
                attn = T.alloc_shared([block_C, block_C], dtype)
                o_frag = T.alloc_fragment([block_C, dim_v], accum_dtype)
                attn_frag = T.alloc_fragment([block_C, block_C], accum_dtype)
                for i, d in T.Parallel(block_C, dim_k):
                    q_c[i, d] = q[bid, base + i, hid, d]
                    k_c[i, d] = k[bid, base + i, hid, d]
                for i in T.Parallel(block_C):
                    g_c[i] = g[bid, base + i, hid]
                T.copy(S[bid, hid, tid, :, :], h_c, disable_tma=True)
                for i, d in T.Parallel(block_C, dim_v):
                    v_new_c[i, d] = v_new[bid, base + i, hid, d]
                T.clear(o_frag)
                T.gemm(q_c, h_c, o_frag)
                for i, j in T.Parallel(block_C, dim_v):
                    o_frag[i, j] = o_frag[i, j] * T.exp2(g_c[i] * LOG2E)
                T.clear(attn_frag)
                T.gemm(q_c, k_c, attn_frag, transpose_B=True)
                for i, j in T.Parallel(block_C, block_C):
                    attn[i, j] = T.if_then_else(i >= j, attn_frag[i, j] * T.exp2((g_c[i] - g_c[j]) * LOG2E), T.float32(0.0))
                T.gemm(attn, v_new_c, o_frag)
                for i, d in T.Parallel(block_C, dim_v):
                    o[bid, base + i, hid, d] = o_frag[i, d]
        return output_o_bthd
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
