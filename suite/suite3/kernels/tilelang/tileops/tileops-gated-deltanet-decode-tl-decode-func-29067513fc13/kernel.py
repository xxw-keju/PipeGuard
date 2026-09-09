# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634
_DEFAULT_K_TILE = 16

def _kernel_factory(batch: int, head: int, dim_k: int, dim_v: int, k_tile: int=_DEFAULT_K_TILE, dtype: str='float32'):
    accum_dtype = 'float32'
    if dim_k % k_tile != 0:
        raise ValueError(f'dim_k={dim_k} must be divisible by k_tile={k_tile}')

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _decode_func(num_stages, threads=128):

        @T.macro
        def _decode_body(q: T.Tensor([batch, head, dim_k], dtype), k: T.Tensor([batch, head, dim_k], dtype), v: T.Tensor([batch, head, dim_v], dtype), g: T.Tensor([batch, head], dtype), beta: T.Tensor([batch, head], dtype), state: T.Tensor([batch, head, dim_k, dim_v], dtype), o: T.Tensor([batch, head, dim_v], dtype), new_state: T.Tensor([batch, head, dim_k, dim_v], dtype)):
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
                    k_val = T.cast(k[bid, hid, kk], accum_dtype)
                    q_val = T.cast(q[bid, hid, kk], accum_dtype)
                    for j in T.Parallel(dim_v):
                        h_val = T.cast(state[bid, hid, kk, j], accum_dtype)
                        sk_frag[j] = sk_frag[j] + k_val * h_val
                        sq_frag[j] = sq_frag[j] + q_val * h_val
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += T.cast(q[bid, hid, kk], accum_dtype) * T.cast(k[bid, hid, kk], accum_dtype)
                for j in T.Parallel(dim_v):
                    v_new[j] = beta_val * T.cast(v[bid, hid, j], accum_dtype) - alpha_beta * sk_frag[j]
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(alpha * sq_frag[j] + qk_dot[0] * v_new[j], dtype)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(alpha * T.cast(h_tile[kk, j], accum_dtype) + T.cast(k[bid, hid, kt * k_tile + kk], accum_dtype) * v_new[j], dtype)

        @T.prim_func
        def gated_deltanet_decode(q: T.Tensor([batch, head, dim_k], dtype), k: T.Tensor([batch, head, dim_k], dtype), v: T.Tensor([batch, head, dim_v], dtype), g: T.Tensor([batch, head], dtype), beta: T.Tensor([batch, head], dtype), state: T.Tensor([batch, head, dim_k, dim_v], dtype), o: T.Tensor([batch, head, dim_v], dtype), new_state: T.Tensor([batch, head, dim_k, dim_v], dtype)):
            _decode_body(q, k, v, g, beta, state, o, new_state)
        return gated_deltanet_decode
    return _decode_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
