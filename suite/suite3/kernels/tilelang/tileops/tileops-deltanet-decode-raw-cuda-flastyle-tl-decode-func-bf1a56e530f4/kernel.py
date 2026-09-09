# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, head: int, dim_k: int, dim_v: int, v_tile: int=16, raw_group_size: int=2, raw_maxrregcount: int=146, dtype: str='bfloat16'):
    if dtype not in ('float16', 'bfloat16'):
        raise ValueError('Raw CUDA DeltaNet decode currently supports float16/bfloat16 only.')
    if dim_k != 128 or dim_v != 128:
        raise ValueError('Raw CUDA DeltaNet decode currently requires DK=DV=128.')
    if dim_v % v_tile != 0:
        raise ValueError(f'dim_v={dim_v} must be divisible by v_tile={v_tile}')
    if raw_group_size != 2:
        raise ValueError('raw_group_size must equal 2 for the fixed two-lane reductions.')
    if raw_group_size * v_tile != 32:
        raise ValueError('raw_group_size * v_tile must equal one warp')
    if dim_k % raw_group_size != 0:
        raise ValueError(f'dim_k={dim_k} must be divisible by raw_group_size={raw_group_size}')
    total_blocks = batch * head * (dim_v // v_tile)
    k_chunk = dim_k // raw_group_size
    raw_compile_flags = ['-O3', '-DENABLE_BF16', '--use_fast_math']
    if raw_maxrregcount > 0:
        raw_compile_flags.append(f'--maxrregcount={raw_maxrregcount}')

    @tilelang.jit(out_idx=[-2, -1], compile_flags=raw_compile_flags)
    def _decode_func(threads=32):

        @T.prim_func
        def deltanet_decode_raw_cuda_flastyle(q: T.Tensor([batch, head, dim_k], dtype), k: T.Tensor([batch, head, dim_k], dtype), v: T.Tensor([batch, head, dim_v], dtype), beta: T.Tensor([batch, head], dtype), state: T.Tensor([batch, head, dim_k, dim_v], dtype), o: T.Tensor([batch, head, dim_v], dtype), new_state: T.Tensor([batch, head, dim_k, dim_v], dtype)):
            with T.Kernel(total_blocks, threads=threads) as (block,):
                tx = T.get_thread_binding()
                vid = block % (dim_v // v_tile)
                nh = block // (dim_v // v_tile)
                bid = nh // head
                hid = nh - bid * head
                v_lane = tx // raw_group_size
                k_rank = tx - v_lane * raw_group_size
                v_idx = vid * v_tile + v_lane
                k_begin = k_rank * k_chunk
                k_shared = T.alloc_shared([dim_k], 'float32')
                q_shared = T.alloc_shared([dim_k], 'float32')
                h = T.alloc_local([k_chunk], 'float32')
                old_partial = T.alloc_var('float32', init=0.0)
                out_partial = T.alloc_var('float32', init=0.0)
                v_new = T.alloc_var('float32', init=0.0)
                for i in T.serial(T.ceildiv(dim_k, 32)):
                    kk = tx + i * 32
                    if kk < dim_k:
                        k_shared[kk] = T.cast(k[bid, hid, kk], 'float32')
                        q_shared[kk] = T.cast(q[bid, hid, kk], 'float32')
                T.sync_threads()
                beta_val = T.cast(beta[bid, hid], 'float32')
                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_val = T.cast(state[bid, hid, kk, v_idx], 'float32')
                    h[ii] = h_val
                    old_partial += h_val * k_shared[kk]
                old_val = old_partial + T.shfl_down(old_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    v_new = beta_val * (T.cast(v[bid, hid, v_idx], 'float32') - old_val)
                v_new = T.shfl_sync(v_new, v_lane * raw_group_size, width=raw_group_size)
                for ii in T.serial(k_chunk):
                    kk = k_begin + ii
                    h_new = h[ii] + k_shared[kk] * v_new
                    new_state[bid, hid, kk, v_idx] = T.cast(h_new, dtype)
                    out_partial += h_new * q_shared[kk]
                out_val = out_partial + T.shfl_down(out_partial, 1, width=raw_group_size)
                if k_rank == 0:
                    o[bid, hid, v_idx] = T.cast(out_val, dtype)
        return deltanet_decode_raw_cuda_flastyle
    return _decode_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
