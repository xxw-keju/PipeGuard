# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, num_chunks: int, n_heads: int, d_state: int, has_initial_states: bool=True, dtype: str='float16') -> Callable:
    accum_dtype = 'float'
    B = batch
    C = num_chunks
    H = n_heads
    D = d_state

    @tilelang.jit(out_idx=[-2, -1])
    def kernel_func(block_d: int, threads: int, vectorize: bool):
        if vectorize:
            assert threads == block_d // 2, f'threads must equal block_d // 2 (got threads={threads}, block_d={block_d})'
        states_shape = (B, C, H, D)
        dA_shape = (B, H, C)
        init_shape = (B, H, D)
        out_shape = (B, C, H, D)
        final_shape = (B, H, D)

        @T.prim_func
        def main(states: T.Tensor(states_shape, dtype), dA_chunk_cumsum: T.Tensor(dA_shape, accum_dtype), initial_states: T.Tensor(init_shape, accum_dtype), out: T.Tensor(out_shape, accum_dtype), final_states: T.Tensor(final_shape, accum_dtype)):
            with T.Kernel(T.ceildiv(D, block_d), B, H, threads=threads) as (bd, bb, bh):
                d0 = bd * block_d
                scale_shared = T.alloc_shared((C,), accum_dtype)
                for c in T.Parallel(C):
                    scale_shared[c] = T.exp(dA_chunk_cumsum[bb, bh, c])
                T.sync_threads()
                if vectorize:
                    s_lo = T.alloc_fragment((threads,), accum_dtype)
                    s_hi = T.alloc_fragment((threads,), accum_dtype)
                    u_lo = T.alloc_fragment((threads,), accum_dtype)
                    u_hi = T.alloc_fragment((threads,), accum_dtype)
                    for i in T.Parallel(threads):
                        di_lo = d0 + i
                        di_hi = d0 + i + threads
                        if has_initial_states:
                            s_lo[i] = T.if_then_else(di_lo < D, initial_states[bb, bh, di_lo], T.float32(0.0))
                            s_hi[i] = T.if_then_else(di_hi < D, initial_states[bb, bh, di_hi], T.float32(0.0))
                        else:
                            s_lo[i] = T.float32(0.0)
                            s_hi[i] = T.float32(0.0)
                    for c in T.serial(C):
                        scale = scale_shared[c]
                        for i in T.Parallel(threads):
                            di_lo = d0 + i
                            di_hi = d0 + i + threads
                            if di_lo < D:
                                out[bb, c, bh, di_lo] = s_lo[i]
                            if di_hi < D:
                                out[bb, c, bh, di_hi] = s_hi[i]
                            u_lo[i] = T.if_then_else(di_lo < D, T.cast(states[bb, c, bh, di_lo], accum_dtype), T.float32(0.0))
                            u_hi[i] = T.if_then_else(di_hi < D, T.cast(states[bb, c, bh, di_hi], accum_dtype), T.float32(0.0))
                        for i in T.Parallel(threads):
                            s_lo[i] = scale * s_lo[i] + u_lo[i]
                            s_hi[i] = scale * s_hi[i] + u_hi[i]
                    for i in T.Parallel(threads):
                        di_lo = d0 + i
                        di_hi = d0 + i + threads
                        if di_lo < D:
                            final_states[bb, bh, di_lo] = s_lo[i]
                        if di_hi < D:
                            final_states[bb, bh, di_hi] = s_hi[i]
                else:
                    s_frag = T.alloc_fragment((block_d,), accum_dtype)
                    u_frag = T.alloc_fragment((block_d,), accum_dtype)
                    for i in T.Parallel(block_d):
                        di = d0 + i
                        if has_initial_states:
                            s_frag[i] = T.if_then_else(di < D, initial_states[bb, bh, di], T.float32(0.0))
                        else:
                            s_frag[i] = T.float32(0.0)
                    for c in T.serial(C):
                        scale = scale_shared[c]
                        for i in T.Parallel(block_d):
                            di = d0 + i
                            if di < D:
                                out[bb, c, bh, di] = s_frag[i]
                            u_frag[i] = T.if_then_else(di < D, T.cast(states[bb, c, bh, di], accum_dtype), T.float32(0.0))
                        for i in T.Parallel(block_d):
                            s_frag[i] = scale * s_frag[i] + u_frag[i]
                    for i in T.Parallel(block_d):
                        di = d0 + i
                        if di < D:
                            final_states[bb, bh, di] = s_frag[i]
        return main
    return kernel_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
