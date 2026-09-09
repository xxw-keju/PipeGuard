# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Any, Callable, Optional
from tilelang import language as T
import tilelang
GEMM_MIN_N = 16
LOG2E: float = 1.4426950408889634

def _kernel_factory(batch: int, seq_len: int, heads: int, dim_k: int, dim_v: int, chunk_size: int, dtype: str, num_v_partitions: int=1, num_k_partitions: int=1) -> Callable:
    """Compute per-chunk hidden states h in forward order.

    Sequential over chunks (inter-chunk recurrence).
    Uses T.Pipelined + T.copy for async prefetch of k, v, g_cumsum.
    g_cumsum is pre-computed — no T.Serial cumsum in this kernel.

    KV-partition parallelism: splits K and V dimensions across thread blocks
    for higher SM utilization and more square GEMM shapes.
    Grid: B * H * num_k_partitions * num_v_partitions blocks.
    """
    accum_dtype = 'float32'
    num_chunks = seq_len // chunk_size
    dim_v_part = dim_v // num_v_partitions
    if dim_v_part < GEMM_MIN_N:
        raise ValueError(f'dim_v ({dim_v}) split across num_v_partitions ({num_v_partitions}) gives a {dim_v_part}-column T.gemm B operand, below the minimum N extent ({GEMM_MIN_N})')
    dim_k_part = dim_k // num_k_partitions
    num_kv = num_k_partitions * num_v_partitions

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True, tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True, tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True})
    def _h_func(num_stages, threads=128):
        k_shape = [batch, seq_len, heads, dim_k]
        v_shape = [batch, seq_len, heads, dim_v]
        g_cumsum_shape = [batch, seq_len, heads, dim_k]
        init_state_shape = [batch, heads, dim_k, dim_v]
        h_out_shape = [batch, num_chunks + 1, heads, dim_k, dim_v]

        @T.prim_func
        def _main(k: T.Tensor(k_shape, dtype), v: T.Tensor(v_shape, dtype), g_cumsum: T.Tensor(g_cumsum_shape, accum_dtype), initial_state: T.Tensor(init_state_shape, accum_dtype), h_out: T.Tensor(h_out_shape, accum_dtype)):
            with T.Kernel(batch * heads * num_kv, threads=threads) as bx:
                i_b = bx // (heads * num_kv)
                i_h = bx // num_kv % heads
                i_kv = bx % num_kv
                i_kp = i_kv // num_v_partitions
                i_vp = i_kv % num_v_partitions
                k_offset = i_kp * dim_k_part
                v_offset = i_vp * dim_v_part
                h_s = T.alloc_shared([dim_k_part, dim_v_part], accum_dtype)
                k_s = T.alloc_shared([chunk_size, dim_k_part], dtype)
                v_s = T.alloc_shared([chunk_size, dim_v_part], dtype)
                g_cumsum_s = T.alloc_shared([chunk_size, dim_k_part], accum_dtype)
                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_s[i_k, i_v] = initial_state[i_b, i_h, k_offset + i_k, v_offset + i_v]
                for i_c in T.Pipelined(num_chunks, num_stages=num_stages):
                    T.copy(k[i_b, i_c * chunk_size:(i_c + 1) * chunk_size, i_h, k_offset:k_offset + dim_k_part], k_s, disable_tma=True)
                    T.copy(v[i_b, i_c * chunk_size:(i_c + 1) * chunk_size, i_h, v_offset:v_offset + dim_v_part], v_s, disable_tma=True)
                    T.copy(g_cumsum[i_b, i_c * chunk_size:(i_c + 1) * chunk_size, i_h, k_offset:k_offset + dim_k_part], g_cumsum_s, disable_tma=True)
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_out[i_b, i_c, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]
                    g_last = T.alloc_fragment([dim_k_part], accum_dtype)
                    for i_k in T.Parallel(dim_k_part):
                        g_last[i_k] = g_cumsum_s[chunk_size - 1, i_k]
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] * T.exp2(g_last[i_k] * LOG2E)
                    k_adj_f = T.alloc_fragment([chunk_size, dim_k_part], dtype)
                    for i_t, i_k in T.Parallel(chunk_size, dim_k_part):
                        k_adj_f[i_t, i_k] = T.cast(T.cast(k_s[i_t, i_k], accum_dtype) * T.exp2((g_last[i_k] - g_cumsum_s[i_t, i_k]) * LOG2E), dtype)
                    delta_h = T.alloc_fragment([dim_k_part, dim_v_part], accum_dtype)
                    T.fill(delta_h, 0.0)
                    T.gemm(k_adj_f, v_s, delta_h, transpose_A=True, policy=T.GemmWarpPolicy.FullRow)
                    for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                        h_s[i_k, i_v] = h_s[i_k, i_v] + delta_h[i_k, i_v]
                for i_k, i_v in T.Parallel(dim_k_part, dim_v_part):
                    h_out[i_b, num_chunks, i_h, k_offset + i_k, v_offset + i_v] = h_s[i_k, i_v]
        return _main
    return _h_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
