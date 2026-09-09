# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634
_DEFAULT_K_TILE = 16

def _kernel_factory(batch: int, head: int, dim_k: int, dim_v: int, k_tile: int=_DEFAULT_K_TILE, scale: float=-1.0):
    """FP32 decode kernel using element-wise matvec instead of T.gemm.

    T.gemm on fp32 inputs uses TF32 tensor cores which truncate the mantissa
    to 10 bits (~1e-3 error per op).  For multi-step decode the error
    compounds through the recurrent state.  This kernel avoids T.gemm
    entirely, computing S@q_gated via scalar accumulation in full fp32.
    """
    dtype = 'float32'
    accum_dtype = 'float32'
    if dim_k % k_tile != 0:
        raise ValueError(f'dim_k={dim_k} must be divisible by k_tile={k_tile}')
    if scale <= 0:
        scale = dim_k ** (-0.5)

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3'])
    def _decode_func(num_stages, threads=128):

        @T.prim_func
        def gla_decode_fp32(q: T.Tensor([batch, head, dim_k], dtype), k: T.Tensor([batch, head, dim_k], dtype), v: T.Tensor([batch, head, dim_v], dtype), gk: T.Tensor([batch, head, dim_k], dtype), state: T.Tensor([batch, head, dim_k, dim_v], dtype), o: T.Tensor([batch, head, dim_v], dtype), new_state: T.Tensor([batch, head, dim_k, dim_v], dtype)):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                q_shared = T.alloc_shared([dim_k], dtype)
                k_shared = T.alloc_shared([dim_k], dtype)
                v_shared = T.alloc_shared([dim_v], dtype)
                gk_shared = T.alloc_shared([dim_k], dtype)
                for i in T.Parallel(dim_k):
                    q_shared[i] = q[bid, hid, i]
                    k_shared[i] = k[bid, hid, i]
                    gk_shared[i] = gk[bid, hid, i]
                for i in T.Parallel(dim_v):
                    v_shared[i] = v[bid, hid, i]
                T.fill(sq_frag, 0.0)
                for kk in T.Serial(dim_k):
                    gk_val = gk_shared[kk]
                    alpha_i = T.exp2(gk_val * LOG2E)
                    q_gated = q_shared[kk] * alpha_i
                    for j in T.Parallel(dim_v):
                        sq_frag[j] = sq_frag[j] + q_gated * state[bid, hid, kk, j]
                qk_dot[0] = 0.0
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j]
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.exp2(gk_shared[kt * k_tile + kk] * LOG2E) * h_tile[kk, j] + k_shared[kt * k_tile + kk] * v_shared[j]
        return gla_decode_fp32
    return _decode_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
