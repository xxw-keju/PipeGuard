# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang
LOG2E: float = 1.4426950408889634
_DEFAULT_K_TILE = 16

def _kernel_factory(batch: int, head: int, dim_k: int, dim_v: int, k_tile: int=_DEFAULT_K_TILE, dtype: str='float32', scale: float=-1.0):
    accum_dtype = 'float32'
    if dim_k % k_tile != 0:
        raise ValueError(f'dim_k={dim_k} must be divisible by k_tile={k_tile}')
    if scale <= 0:
        scale = dim_k ** (-0.5)

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _decode_func(num_stages, threads=128):

        @T.prim_func
        def gla_decode(q: T.Tensor([batch, head, dim_k], dtype), k: T.Tensor([batch, head, dim_k], dtype), v: T.Tensor([batch, head, dim_v], dtype), gk: T.Tensor([batch, head, dim_k], dtype), state: T.Tensor([batch, head, dim_k, dim_v], dtype), o: T.Tensor([batch, head, dim_v], dtype), new_state: T.Tensor([batch, head, dim_k, dim_v], dtype)):
            with T.Kernel(batch, head, threads=threads) as (bid, hid):
                h_tile = T.alloc_shared([k_tile, dim_v], dtype)
                h_tile_o = T.alloc_shared([k_tile, dim_v], dtype)
                v_shared = T.alloc_shared([dim_v], accum_dtype)
                sq_frag = T.alloc_fragment([dim_v], accum_dtype)
                qk_dot = T.alloc_local([1], accum_dtype)
                q_shared = T.alloc_shared([dim_k], accum_dtype)
                k_shared = T.alloc_shared([dim_k], accum_dtype)
                gk_shared = T.alloc_shared([dim_k], accum_dtype)
                for j in T.Parallel(dim_v):
                    v_shared[j] = T.cast(v[bid, hid, j], accum_dtype)
                for i in T.Parallel(dim_k):
                    q_shared[i] = T.cast(q[bid, hid, i], accum_dtype)
                    k_shared[i] = T.cast(k[bid, hid, i], accum_dtype)
                    gk_shared[i] = T.cast(gk[bid, hid, i], accum_dtype)
                T.fill(sq_frag, 0.0)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile_o)
                    for kk in T.Serial(k_tile):
                        gk_val = gk_shared[kt * k_tile + kk]
                        alpha_i = T.exp2(gk_val * LOG2E)
                        q_gated = q_shared[kt * k_tile + kk] * alpha_i
                        for j in T.Parallel(dim_v):
                            sq_frag[j] = sq_frag[j] + q_gated * T.cast(h_tile_o[kk, j], accum_dtype)
                qk_dot[0] = T.float32(0.0)
                for kk in T.Serial(dim_k):
                    qk_dot[0] += q_shared[kk] * k_shared[kk]
                for j in T.Parallel(dim_v):
                    o[bid, hid, j] = T.cast(scale * sq_frag[j] + scale * qk_dot[0] * v_shared[j], dtype)
                for kt in T.Pipelined(dim_k // k_tile, num_stages=num_stages):
                    T.copy(state[bid, hid, kt * k_tile, 0], h_tile)
                    for kk, j in T.Parallel(k_tile, dim_v):
                        new_state[bid, hid, kt * k_tile + kk, j] = T.cast(T.exp2(gk_shared[kt * k_tile + kk] * LOG2E) * T.cast(h_tile[kk, j], accum_dtype) + k_shared[kt * k_tile + kk] * v_shared[j], dtype)
        return gla_decode
    return _decode_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
