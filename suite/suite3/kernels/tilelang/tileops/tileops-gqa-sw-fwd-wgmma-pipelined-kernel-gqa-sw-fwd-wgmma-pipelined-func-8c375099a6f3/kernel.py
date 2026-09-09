# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional, Tuple
import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634

def make_apply_softcap(score_scale, softcap, accum_dtype, block_rows, block_cols):
    """Create a T.macro that applies score softcap before online softmax.

    The input score fragment contains raw QK scores plus mask entries. Valid
    scores are transformed as ``softcap * tanh((score * score_scale) / softcap)``.
    Masked ``-inf`` entries are preserved so they remain invisible to softmax.
    """

    @T.macro
    def apply_softcap(acc_s):
        for i, j in T.Parallel(block_rows, block_cols):
            capped = T.cast(softcap, accum_dtype) * T.tanh(acc_s[i, j] * T.cast(score_scale / softcap, accum_dtype))
            acc_s[i, j] = T.if_then_else(acc_s[i, j] == -T.infinity(accum_dtype), -T.infinity(accum_dtype), capped)
    return apply_softcap

def _kernel_factory(batch: int, heads: int, heads_kv: int, seq_len: int, dim: int, is_causal: bool, window_size_left: int, window_size_right: int, sm_scale: Optional[float]=None, softcap: float=0.0, dtype: str='float16') -> Callable:
    score_scale = dim ** (-0.5) if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    groups = heads // heads_kv
    accum_dtype = 'float'
    has_window = window_size_left >= 0 or window_size_right >= 0

    @tilelang.jit(out_idx=[3, 4], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _gqa_sw_fwd_wgmma_pipelined_func(block_m, block_n, num_stages, threads):
        q_shape = (batch, seq_len, heads, dim)
        kv_shape = (batch, seq_len, heads_kv, dim)
        apply_softcap = make_apply_softcap(score_scale, softcap, accum_dtype, block_m, block_n) if use_softcap else None

        @T.macro
        def mma0(k: T.Tensor(kv_shape, dtype), q_shared: T.SharedBuffer([block_m, dim], dtype), k_shared: T.SharedBuffer([block_n, dim], dtype), acc_s: T.FragmentBuffer([block_m, block_n], accum_dtype), k_idx: T.int32, bx: T.int32, by: T.int32, bz: T.int32) -> None:
            T.copy(k[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :], k_shared)
            if is_causal and has_window:
                for i, j in T.Parallel(block_m, block_n):
                    causal_mask = bx * block_m + i < k_idx * block_n + j
                    left_mask = window_size_left >= 0 and k_idx * block_n + j < bx * block_m + i - window_size_left
                    acc_s[i, j] = T.if_then_else(causal_mask or left_mask, -T.infinity(accum_dtype), 0)
            elif is_causal:
                for i, j in T.Parallel(block_m, block_n):
                    acc_s[i, j] = T.if_then_else(bx * block_m + i < k_idx * block_n + j, -T.infinity(accum_dtype), 0)
            elif has_window:
                for i, j in T.Parallel(block_m, block_n):
                    left_mask = window_size_left >= 0 and k_idx * block_n + j < bx * block_m + i - window_size_left
                    right_mask = window_size_right >= 0 and k_idx * block_n + j > bx * block_m + i + window_size_right
                    acc_s[i, j] = T.if_then_else(left_mask or right_mask, -T.infinity(accum_dtype), 0)
            else:
                T.clear(acc_s)
            T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)

        @T.macro
        def mma1(v: T.Tensor(kv_shape, dtype), v_shared: T.SharedBuffer([block_n, dim], dtype), acc_s_cast: T.FragmentBuffer([block_m, block_n], dtype), acc_o: T.FragmentBuffer([block_m, dim], accum_dtype), k_idx: T.int32, by: T.int32, bz: T.int32) -> None:
            T.copy(v[bz, k_idx * block_n:(k_idx + 1) * block_n, by // groups, :], v_shared)
            T.gemm(acc_s_cast, v_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)

        @T.prim_func
        def _gqa_sw_fwd_wgmma_pipelined_main(q: T.Tensor(q_shape, dtype), k: T.Tensor(kv_shape, dtype), v: T.Tensor(kv_shape, dtype), output: T.Tensor(q_shape, dtype), lse: T.Tensor([batch, heads, seq_len], accum_dtype)) -> None:
            with T.Kernel(T.ceildiv(seq_len, block_m), heads, batch, threads=threads) as (bx, by, bz):
                q_shared = T.alloc_shared([block_m, dim], dtype)
                k_shared = T.alloc_shared([block_n, dim], dtype)
                v_shared = T.alloc_shared([block_n, dim], dtype)
                o_shared = T.alloc_shared([block_m, dim], dtype)
                acc_s = T.alloc_fragment([block_m, block_n], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_m, block_n], dtype)
                acc_o = T.alloc_fragment([block_m, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_m], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_m], accum_dtype)
                scores_scale = T.alloc_fragment([block_m], accum_dtype)
                scores_sum = T.alloc_fragment([block_m], accum_dtype)
                logsum = T.alloc_fragment([block_m], accum_dtype)
                T.annotate_layout({o_shared: tilelang.layout.make_swizzled_layout(o_shared)})
                T.copy(q[bz, bx * block_m:(bx + 1) * block_m, by, :], q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))
                if is_causal:
                    k_end = T.ceildiv(T.min(seq_len, (bx + 1) * block_m), block_n)
                elif has_window and window_size_right >= 0:
                    k_end = T.ceildiv(T.min(seq_len, (bx + 1) * block_m + window_size_right), block_n)
                else:
                    k_end = T.ceildiv(seq_len, block_n)
                if has_window and window_size_left >= 0:
                    k_start = T.max(0, bx * block_m - window_size_left) // block_n
                else:
                    k_start = 0
                loop_count = T.max(k_end - k_start, 0)
                for k_offset in T.Pipelined(loop_count, num_stages=num_stages):
                    k_idx = k_start + k_offset
                    mma0(k, q_shared, k_shared, acc_s, k_idx, bx, by, bz)
                    if use_softcap:
                        apply_softcap(acc_s)
                    T.copy(scores_max, scores_max_prev)
                    T.fill(scores_max, -T.infinity(accum_dtype))
                    T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    for i in T.Parallel(block_m):
                        scores_max[i] = T.max(scores_max[i], T.cast(-1e+38, accum_dtype))
                    for i in T.Parallel(block_m):
                        scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                    for i, j in T.Parallel(block_m, block_n):
                        acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                    T.reduce_sum(acc_s, scores_sum, dim=1)
                    for i in T.Parallel(block_m):
                        logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                    T.copy(acc_s, acc_s_cast)
                    for i, j in T.Parallel(block_m, dim):
                        acc_o[i, j] *= scores_scale[i]
                    mma1(v, v_shared, acc_s_cast, acc_o, k_idx, by, bz)
                for i, j in T.Parallel(block_m, dim):
                    acc_o[i, j] /= logsum[i]
                T.sync_threads(3, threads)
                T.copy(acc_o, o_shared)
                T.sync_threads(3, threads)
                T.copy(o_shared, output[bz, bx * block_m:(bx + 1) * block_m, by, :])
                for i in T.Parallel(block_m):
                    logsum[i] = T.log2(logsum[i]) + scores_max[i] * scale
                T.copy(logsum, lse[bz, by, bx * block_m:(bx + 1) * block_m])
        return _gqa_sw_fwd_wgmma_pipelined_main
    return _gqa_sw_fwd_wgmma_pipelined_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
