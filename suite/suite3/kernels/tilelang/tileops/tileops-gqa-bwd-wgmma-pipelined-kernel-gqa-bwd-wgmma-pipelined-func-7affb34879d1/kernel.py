# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional, Tuple
import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634

def _kernel_factory(batch: int, heads: int, heads_kv: int, seq_len: int, dim: int, is_causal: bool, dtype: str='float16') -> Callable:
    sm_scale = (1.0 / dim) ** 0.5
    scale = (1.0 / dim) ** 0.5 * LOG2E
    if heads % heads_kv != 0:
        raise ValueError('heads must be divisible by heads_kv')
    groups = heads // heads_kv
    accum_dtype = 'float'

    @tilelang.jit(pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _gqa_bwd_wgmma_pipelined_func(block_m: int, block_n: int, num_stages: int, threads: int) -> Callable:
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)

        @T.prim_func
        def _gqa_bwd_wgmma_pipelined_main(q: T.Tensor(q_shape, dtype), k: T.Tensor(kv_shape, dtype), v: T.Tensor(kv_shape, dtype), do: T.Tensor(q_shape, dtype), lse: T.Tensor([batch, heads, seq_len], accum_dtype), delta: T.Tensor([batch, heads, seq_len], accum_dtype), dq: T.Tensor(q_shape, accum_dtype), dk: T.Tensor(kv_shape, accum_dtype), dv: T.Tensor(kv_shape, accum_dtype)) -> None:
            with T.Kernel(heads, T.ceildiv(seq_len, block_m), batch, threads=threads) as (bx, by, bz):
                k_shared = T.alloc_shared([block_m, dim], dtype)
                dst_shared = T.alloc_shared([block_m, block_n], dtype)
                q_frag = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_m, dim], dtype)
                qkt = T.alloc_fragment([block_m, block_n], accum_dtype)
                dst = T.alloc_fragment([block_m, block_n], accum_dtype)
                qkt_cast = T.alloc_fragment([block_m, block_n], dtype)
                dst_cast = T.alloc_fragment([block_m, block_n], dtype)
                lse_shared = T.alloc_shared([block_n], accum_dtype)
                delta_shared = T.alloc_shared([block_n], accum_dtype)
                do_shared = T.alloc_shared([block_n, dim], dtype)
                dv_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dk_frag = T.alloc_fragment([block_m, dim], accum_dtype)
                dq_frag = T.alloc_fragment([block_n, dim], accum_dtype)
                dq_shared = T.alloc_shared([block_n, dim], accum_dtype)
                dv_shared = T.alloc_shared([block_m, dim], accum_dtype)
                dk_shared = T.alloc_shared([block_m, dim], accum_dtype)
                T.annotate_layout({dq_shared: tilelang.layout.make_swizzled_layout(dq_shared), dv_shared: tilelang.layout.make_swizzled_layout(dv_shared), dk_shared: tilelang.layout.make_swizzled_layout(dk_shared)})
                T.copy(k[bz, by * block_m:(by + 1) * block_m, bx // groups, :], k_shared)
                T.copy(v[bz, by * block_m:(by + 1) * block_m, bx // groups, :], v_shared)
                T.clear(dv_frag)
                T.clear(dk_frag)
                loop_st = T.floordiv(by * block_m, block_n) if is_causal else 0
                loop_ed = T.ceildiv(seq_len, block_n)
                for k_idx in T.Pipelined(loop_st, loop_ed, num_stages=num_stages):
                    T.copy(q[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], q_frag)
                    T.clear(qkt)
                    T.wgmma_gemm(k_shared, q_frag, qkt, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(lse[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], lse_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        qkt[i, j] = T.exp2(qkt[i, j] * scale - lse_shared[j])
                    if is_causal:
                        for i, j in T.Parallel(block_m, block_n):
                            qkt[i, j] = T.if_then_else(by * block_m + i <= k_idx * block_n + j, qkt[i, j], 0)
                    T.copy(do[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], do_shared)
                    T.clear(dst)
                    T.wgmma_gemm(v_shared, do_shared, dst, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(1)
                    T.copy(qkt, qkt_cast)
                    T.wgmma_gemm(qkt_cast, do_shared, dv_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(delta[bz, bx, k_idx * block_n:(k_idx + 1) * block_n], delta_shared)
                    for i, j in T.Parallel(block_m, block_n):
                        dst_cast[i, j] = qkt[i, j] * (dst[i, j] - delta_shared[j]) * sm_scale
                    T.wait_wgmma(0)
                    T.wgmma_gemm(dst_cast, q_frag, dk_frag, policy=T.GemmWarpPolicy.FullRow)
                    T.wait_wgmma(1)
                    T.copy(dst_cast, dst_shared)
                    T.clear(dq_frag)
                    T.wgmma_gemm(dst_shared, k_shared, dq_frag, transpose_A=True)
                    T.wait_wgmma(1)
                    T.wait_wgmma(0)
                    T.copy(dq_frag, dq_shared)
                    T.atomic_add(dq[bz, k_idx * block_n:(k_idx + 1) * block_n, bx, :], dq_shared)
                T.copy(dv_frag, dv_shared)
                T.atomic_add(dv[bz, by * block_m:(by + 1) * block_m, bx // groups, :], dv_shared)
                T.copy(dk_frag, dk_shared)
                T.atomic_add(dk[bz, by * block_m:(by + 1) * block_m, bx // groups, :], dk_shared)
        return _gqa_bwd_wgmma_pipelined_main
    return _gqa_bwd_wgmma_pipelined_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
