# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
_DEFAULT_K_TILE = 16

def _kernel_factory(batch: int, head: int, dim_k: int, dim_v: int, k_tile: int=_DEFAULT_K_TILE):
    """FP32 decode kernel using element-wise matvec instead of T.gemm.

    T.gemm on fp32 inputs uses TF32 tensor cores which truncate the mantissa
    to 10 bits (~1e-3 error per op).  For multi-step decode the error
    compounds through the recurrent state.  This kernel avoids T.gemm
    entirely, computing S@k and S@q via scalar accumulation in full fp32.
    """
    dtype = 'float32'
    accum_dtype = 'float32'
    if dim_k % k_tile != 0:
        raise ValueError(f'dim_k={dim_k} must be divisible by k_tile={k_tile}')

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3'])
    def _decode_func(num_stages, threads=128):

        @T.prim_func
        def deltanet_decode_fp32(q: T.Tensor([batch, head, dim_k], dtype), k: T.Tensor([batch, head, dim_k], dtype), v: T.Tensor([batch, head, dim_v], dtype), beta: T.Tensor([batch, head], dtype), state: T.Tensor([batch, head, dim_k, dim_v], dtype), o: T.Tensor([batch, head, dim_v], dtype), new_state: T.Tensor([batch, head, dim_k, dim_v], dtype)):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                sk_frag = T.alloc_fragment([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                v_new = T.alloc_shared([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                beta_val = T.cast(beta[bid, hid], accum_dtype)
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
                    v_new[j] = beta_val * (v[bid, hid, j] - sk_frag[j])
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = sq_frag[j] + qk_dot[0] * v_new[j]
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = h_tile[kk, j] + k[bid, hid, kt * k_tile + kk] * v_new[j]
        return deltanet_decode_fp32
    return _decode_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
