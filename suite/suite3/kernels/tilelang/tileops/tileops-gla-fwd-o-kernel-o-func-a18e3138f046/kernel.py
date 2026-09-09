# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Any, Callable, Optional
from tilelang import language as T
import tilelang
LOG2E: float = 1.4426950408889634

def _kernel_factory(batch: int, seq_len: int, heads: int, dim_k: int, dim_v: int, chunk_size: int, scale: float, dtype: str) -> Callable:
    """Compute output o for each chunk independently.

    Parallel over (batch, heads, chunks): B*H*NC thread blocks.
    Each block reads h[i_c] and g_cumsum from global memory.
    """
    accum_dtype = 'float32'
    num_chunks = seq_len // chunk_size

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def _o_func(num_stages, threads=128):
        q_shape = [batch, seq_len, heads, dim_k]
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        h_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]
        o_shape = [batch, seq_len, heads, dim_v]

        @T.prim_func
        def _main(q: T.Tensor(q_shape, dtype), k: T.Tensor(k_shape, dtype), v: T.Tensor(v_shape, dtype), g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype), h: T.Tensor(h_shape, accum_dtype), o: T.Tensor(o_shape, dtype)):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = bx // num_chunks % heads
                i_c = bx % num_chunks
                chunk_start = i_c * chunk_size
                h_cast_s = T.alloc_shared([dim_k, dim_v], dtype)
                q_s = T.alloc_shared([chunk_size, dim_k], dtype)
                k_s = T.alloc_shared([chunk_size, dim_k], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                q_gated_s = T.alloc_shared([chunk_size, dim_k], dtype)
                A_s = T.alloc_shared([chunk_size, chunk_size], dtype)
                T.copy(q[i_b, chunk_start:chunk_start + chunk_size, i_h, :], q_s, disable_tma=True)
                T.copy(k[i_b, chunk_start:chunk_start + chunk_size, i_h, :], k_s, disable_tma=True)
                T.copy(v[i_b, chunk_start:chunk_start + chunk_size, i_h, :], v_s, disable_tma=True)
                T.copy(g_cumsum[i_b, chunk_start:chunk_start + chunk_size, i_h, :], g_cumsum_s, disable_tma=True)
                for i_k, i_v in T.Parallel(dim_k, dim_v):
                    h_cast_s[i_k, i_v] = T.cast(h[i_b, i_c, i_h, i_k, i_v], dtype)
                for i_t, i_k in T.Parallel(chunk_size, dim_k):
                    q_gated_s[i_t, i_k] = T.cast(T.cast(q_s[i_t, i_k], accum_dtype) * T.exp2(g_cumsum_s[i_t, i_k] * LOG2E), dtype)
                A_frag = T.alloc_fragment([chunk_size, chunk_size], accum_dtype)
                T.fill(A_frag, 0.0)
                for i_k in T.Serial(dim_k):
                    for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                        A_frag[i_t, i_j] = A_frag[i_t, i_j] + T.cast(q_s[i_t, i_k], accum_dtype) * T.cast(k_s[i_j, i_k], accum_dtype) * T.exp2((g_cumsum_s[i_t, i_k] - g_cumsum_s[i_j, i_k]) * LOG2E)
                for i_t, i_j in T.Parallel(chunk_size, chunk_size):
                    A_s[i_t, i_j] = T.cast(T.if_then_else(i_j <= i_t, A_frag[i_t, i_j] * scale, 0.0), dtype)
                acc = T.alloc_fragment([chunk_size, dim_v], accum_dtype)
                T.fill(acc, 0.0)
                T.gemm(q_gated_s, h_cast_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    acc[i_t, i_v] = acc[i_t, i_v] * scale
                T.gemm(A_s, v_s, acc, policy=T.GemmWarpPolicy.FullRow)
                for i_t, i_v in T.Parallel(chunk_size, dim_v):
                    o[i_b, chunk_start + i_t, i_h, i_v] = T.cast(acc[i_t, i_v], dtype)
        return _main
    return _o_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
