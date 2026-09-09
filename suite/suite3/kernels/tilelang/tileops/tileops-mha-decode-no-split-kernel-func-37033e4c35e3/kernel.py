# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634

def make_log2e_scale(dim):
    """Compute softmax scale with log2(e) factor pre-multiplied.

    Returns (1/sqrt(dim)) * log2(e) for use with exp2-based softmax.
    """
    return (1.0 / dim) ** 0.5 * LOG2E

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

def _kernel_factory(batch, heads, seqlen_q, seqlen_kv, dim, page_size, is_causal, dtype):
    scale = make_log2e_scale(dim)
    accum_dtype = 'float'

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(block_M, block_N, num_stages, threads):
        shape_q = [batch, seqlen_q, heads, dim]
        shape_kv = [seqlen_kv, heads, dim]
        online_softmax = make_online_softmax(scale, accum_dtype, block_M, block_N)
        rescale = make_rescale(block_M, dim)

        @T.prim_func
        def mha_decode_no_split(Q: T.Tensor(shape_q, dtype), K: T.Tensor(shape_kv, dtype), V: T.Tensor(shape_kv, dtype), real_seqlen_kv: T.Tensor([batch], T.int32), block_table: T.Tensor([batch, seqlen_kv // page_size], T.int32), Output: T.Tensor(shape_q, dtype)):
            with T.Kernel(T.ceildiv(seqlen_q, block_M), heads, batch, threads=threads) as (bx, by, bz):
                seqlen_kv = real_seqlen_kv[bz]
                Q_shared = T.alloc_shared([block_M, dim], dtype)
                K_shared = T.alloc_shared([block_N, dim], dtype)
                V_shared = T.alloc_shared([block_N, dim], dtype)
                acc_s = T.alloc_fragment([block_M, block_N], accum_dtype)
                acc_s_cast = T.alloc_fragment([block_M, block_N], dtype)
                acc_o = T.alloc_fragment([block_M, dim], accum_dtype)
                scores_max = T.alloc_fragment([block_M], accum_dtype)
                scores_max_prev = T.alloc_fragment([block_M], accum_dtype)
                scores_scale = T.alloc_fragment([block_M], accum_dtype)
                scores_sum = T.alloc_fragment([block_M], accum_dtype)
                logsum = T.alloc_fragment([block_M], accum_dtype)
                T.copy(Q[bz, bx * block_M:(bx + 1) * block_M, by, :], Q_shared)
                T.clear(acc_o)
                T.clear(logsum)
                T.fill(scores_max, -T.infinity(accum_dtype))
                loop_range = T.ceildiv((bx + 1) * block_M, block_N) if is_causal else T.ceildiv(seqlen_kv, block_N)
                num_blockn_in_page = page_size // block_N
                for k in T.Pipelined(loop_range, num_stages=num_stages):
                    page_idx = k // num_blockn_in_page
                    block_idx_in_page = k % num_blockn_in_page
                    blockn_num_offset = block_table[bz, page_idx] * num_blockn_in_page + block_idx_in_page
                    T.copy(K[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N, by, :], K_shared)
                    if is_causal:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(bx * block_M + i >= k * block_N + j, 0, -T.infinity(acc_s.dtype))
                    else:
                        for i, j in T.Parallel(block_M, block_N):
                            acc_s[i, j] = T.if_then_else(k * block_N + j < seqlen_kv, 0, -T.infinity(acc_s.dtype))
                    T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                    T.copy(V[blockn_num_offset * block_N:(blockn_num_offset + 1) * block_N, by, :], V_shared)
                    online_softmax(acc_s, scores_max, scores_max_prev, scores_scale, scores_sum, logsum)
                    T.copy(acc_s, acc_s_cast)
                    rescale(acc_o, scores_scale)
                    T.gemm(acc_s_cast, V_shared, acc_o, policy=T.GemmWarpPolicy.FullRow)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] = T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i])
                T.copy(acc_o, Output[bz, bx * block_M:(bx + 1) * block_M, by, :])
                for i in T.Parallel(block_M):
                    logsum_safe = T.if_then_else(logsum[i] == 0, 1, logsum[i])
                    logsum[i] = T.log2(logsum_safe) + scores_max[i] * scale
        return mha_decode_no_split
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
