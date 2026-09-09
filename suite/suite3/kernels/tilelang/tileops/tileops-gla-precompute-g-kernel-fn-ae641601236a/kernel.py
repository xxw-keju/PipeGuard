# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Any, Callable, Optional
from tilelang import language as T
import tilelang

def _kernel_factory(batch: int, seq_len: int, heads: int, dim_k: int, chunk_size: int, dtype: str) -> Callable:
    """Pre-compute intra-chunk cumulative sum of g.

    Parallel over (batch, heads, chunks): B*H*NC thread blocks.
    Each block computes cumsum for one chunk independently.
    """
    accum_dtype = 'float32'
    num_chunks = seq_len // chunk_size

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def _fn(num_stages, threads=128):
        g_shape = [batch, seq_len, heads, dim_k]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]

        @T.prim_func
        def _main(g: T.Tensor(g_shape, dtype), g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype)):
            with T.Kernel(batch * heads * num_chunks, threads=threads) as bx:
                i_b = bx // (heads * num_chunks)
                i_h = bx // num_chunks % heads
                i_c = bx % num_chunks
                cs = i_c * chunk_size
                g_s = T.alloc_shared([chunk_size, dim_k], dtype)
                g_out_s = T.alloc_shared([chunk_size, dim_k], accum_dtype)
                T.copy(g[i_b, cs:cs + chunk_size, i_h, :], g_s, disable_tma=True)
                for i_k in T.Parallel(dim_k):
                    g_out_s[0, i_k] = T.cast(g_s[0, i_k], accum_dtype)
                for i_t in T.Serial(1, chunk_size):
                    for i_k in T.Parallel(dim_k):
                        g_out_s[i_t, i_k] = g_out_s[i_t - 1, i_k] + T.cast(g_s[i_t, i_k], accum_dtype)
                T.copy(g_out_s, g_cumsum[i_b, cs:cs + chunk_size, i_h, :])
        return _main
    return _fn

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
