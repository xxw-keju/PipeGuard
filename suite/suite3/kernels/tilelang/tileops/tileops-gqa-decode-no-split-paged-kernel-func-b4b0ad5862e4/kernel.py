# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

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

def make_online_softmax(scale, accum_dtype, block_rows, block_cols):
    """Create a reusable online softmax T.macro.

    The macro computes the online softmax update step:
        1. Save previous scores_max and reset to -inf
        2. Reduce max over current scores
        3. Compute rescaling factor for previous accumulator
        4. Apply exp2 scaling to current scores
        5. Reduce sum and update logsum

    Note: The caller is responsible for casting acc_s afterwards
    (e.g., ``T.copy(acc_s, acc_s_cast)``) and rescaling acc_o.

    Use ``make_online_softmax_with_mask_guard`` instead when any attention block
    may be fully masked (all scores = -infinity), e.g. sliding-window or varlen
    attention with OOB guards.

    Args:
        scale: Pre-computed scale factor (should include LOG2E multiplication)
        accum_dtype: Accumulator data type (e.g., "float")
        block_rows: Number of rows in the score matrix
        block_cols: Number of columns in the score matrix

    Returns:
        online_softmax: A T.macro that performs the online softmax update
    """

    @T.macro
    def online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        for i in T.Parallel(block_rows):
            scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
        for i, j in T.Parallel(block_rows, block_cols):
            acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
        T.reduce_sum(acc_s, scores_sum, dim=1)
        for i in T.Parallel(block_rows):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
    return online_softmax

def make_rescale(block_rows, head_dim):
    """Create a reusable rescale T.macro for the output accumulator.

    Args:
        block_rows: Number of rows in the output accumulator
        head_dim: Dimension of each head

    Returns:
        rescale: A T.macro that rescales acc_o by scores_scale
    """

    @T.macro
    def rescale(acc_o, scores_scale):
        for i, j in T.Parallel(block_rows, head_dim):
            acc_o[i, j] *= scores_scale[i]
    return rescale

def _kernel_factory(batch, heads, groups, seqlen_kv, dim, page_size, sm_scale, softcap, dtype):
    score_scale = dim ** (-0.5) if sm_scale is None else sm_scale
    use_softcap = softcap > 0.0
    scale = LOG2E if use_softcap else score_scale * LOG2E
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(block_H, block_N, num_stages, threads):
        shape_q = [batch, heads, dim]
        shape_kv = [seqlen_kv, groups, dim]
        kv_group_num = heads // groups
        valid_block_H = min(block_H, kv_group_num)
        online_softmax = make_online_softmax(scale, accum_dtype, block_H, block_N)
        apply_softcap = make_apply_softcap(score_scale, softcap, accum_dtype, block_H, block_N) if use_softcap else None
        rescale = make_rescale(block_H, dim)

        @T.prim_func
        def gqa_decode_no_split(Q: T.Tensor(shape_q, dtype), K: T.Tensor(shape_kv, dtype), V: T.Tensor(shape_kv, dtype), real_seqlen_kv: T.Tensor([batch], T.int32), block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32), Output: T.Tensor([batch, heads, dim], dtype)):
            with T.Kernel(batch, heads // valid_block_H, 1, threads=threads) as (bx, by, bz):
                Q_shared = T.alloc_shared([block_H, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                O_shared = T.alloc_shared([valid_block_H, dim], dtype)
                acc_s = T.alloc_fragment([block_H, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_H, block_N], dtype)
                acc_o = T.alloc_fragment([block_H, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_H], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_H], accum_dtype)
                scores_scale = T.alloc_fragment([block_H], accum_dtype)
                scores_sum = T.alloc_fragment([block_H], accum_dtype)
                logsum = T.alloc_fragment([block_H], accum_dtype)
                bid = bx
                hid = by
                cur_kv_head = hid // (kv_group_num // valid_block_H)
                seqlen_kv_b = real_seqlen_kv[bid]
                T.copy(Q[bid, hid * valid_block_H:hid * valid_block_H + block_H, :], Q_shared)
                T.fill(acc_o, 0)
                T.fill(logsum, 0)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = T.ceildiv(seqlen_kv_b, block_N)
                num_blockn_in_page = page_size // block_N
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    page_idx = k // num_blockn_in_page
                    block_idx_in_page = k % num_blockn_in_page
                    blockn_num_offset = block_table[bid, page_idx] * num_blockn_in_page + block_idx_in_page
                    T.copy(K[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N, cur_kv_head, :], K_shared)
                    T.copy(V[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N, cur_kv_head, :], V_shared)
                    T.clear(acc_s)
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    for i, j in T.Parallel(block_H, block_N):
                        acc_s[i, j] = T.if_then_else(k * block_N + j < seqlen_kv_b, acc_s[i, j], -T.infinity(accum_dtype))
                    if use_softcap:
                        apply_softcap(acc_s)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_H, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                for i in T.Parallel(block_H):
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale
                T.copy(acc_o[:valid_block_H, :], O_shared)
                T.copy(O_shared, Output[bid, hid * valid_block_H:(hid + 1) * valid_block_H, :])
        return gqa_decode_no_split
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
