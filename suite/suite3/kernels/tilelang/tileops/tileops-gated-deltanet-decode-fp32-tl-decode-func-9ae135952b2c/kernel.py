# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634
_DEFAULT_K_TILE = 16

def _kernel_factory(batch: int, head: int, dim_k: int, dim_v: int, k_tile: int=_DEFAULT_K_TILE):
    """FP32 decode kernel using element-wise matvec instead of T.gemm.

    T.gemm on fp32 inputs uses TF32 tensor cores which truncate the mantissa
    to 10 bits (~1e-3 error per op).  For multi-step decode the error
    compounds through the recurrent state.  This kernel avoids T.gemm
    entirely, computing S@k and S@q via scalar accumulation in full fp32.

    Pass 1 (matvec) uses T.Serial with fragment accumulators and direct
    global memory reads.  The serial k-dimension loop is intentional:
    it ensures scalar multiply-accumulate (no TF32 tensor cores) at the
    cost of lower throughput — acceptable for single-step decode latency.
    Pass 2 (state update) uses T.Pipelined since each tile writes to
    independent global memory.
    """
    dtype = 'float32'
    accum_dtype = 'float32'
    if dim_k % k_tile != 0:
        raise ValueError(f'dim_k={dim_k} must be divisible by k_tile={k_tile}')

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3'])
    def _decode_func(num_stages, threads=128):

        @T.prim_func
        def gated_deltanet_decode_fp32(q: T.Tensor([batch, head, dim_k], dtype), k: T.Tensor([batch, head, dim_k], dtype), v: T.Tensor([batch, head, dim_v], dtype), g: T.Tensor([batch, head], dtype), beta: T.Tensor([batch, head], dtype), state: T.Tensor([batch, head, dim_k, dim_v], dtype), o: T.Tensor([batch, head, dim_v], dtype), new_state: T.Tensor([batch, head, dim_k, dim_v], dtype)):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                sk_frag = T.alloc_fragment([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                g_val = T.cast(g[bid, hid], accum_dtype)
                beta_val = T.cast(beta[bid, hid], accum_dtype)
                alpha = T.exp2(g_val * LOG2E)
                alpha_beta = alpha * beta_val
                T.fill(sk_frag, 0.0)
                T.fill(sq_frag, 0.0)
                for kk in T.Serial(dim_k):
                    k_val = k[bid, hid, kk]
                    q_val = q[bid, hid, kk]
                    for j in T.Parallel(dim_v):
                        h_val = state[bid, hid, kk, j]
                        sk_frag[j] = sk_frag[j] + k_val * h_val
                        sq_frag[j] = sq_frag[j] + q_val * h_val
                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q[bid, hid, kk] * k[bid, hid, kk]
                for j in T.Parallel(dim_v):
                    v_new[j] = beta_val * v[bid, hid, j] - alpha_beta * sk_frag[j]
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = alpha * sq_frag[j] + qk_dot[0] * v_new[j]
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = alpha * h_tile[kk, j] + k[bid, hid, kt * k_tile + kk] * v_new[j]
        return gated_deltanet_decode_fp32
    return _decode_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
