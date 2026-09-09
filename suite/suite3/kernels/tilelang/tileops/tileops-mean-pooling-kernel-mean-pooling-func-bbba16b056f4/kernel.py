# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Any, Callable, Optional
import tilelang.language as T
import tilelang

def _kernel_factory(batch_size: int, seq_len: int, heads: int, dim: int, chunk_size: int, chunks_per_batch: int, seq_num: int, use_offsets: int, dtype: str, accum_dtype: str) -> Callable:

    @tilelang.jit(out_idx=[1])
    def _mean_pooling_func(bdim: int, threads: int) -> None:

        @T.prim_func
        def _mean_pooling_main(x: T.Tensor((batch_size, seq_len, heads, dim), dtype), o: T.Tensor((batch_size, chunks_per_batch, heads, dim), dtype), offsets: T.Tensor((seq_num + 1,), T.int32), indices: T.Tensor((chunks_per_batch, 2), T.int32)) -> None:
            with T.Kernel(T.ceildiv(dim, bdim), chunks_per_batch, batch_size * heads, threads=threads) as (i_d, i_t, i_bh):
                i_b = i_bh // heads
                i_h = i_bh % heads
                x_shared = T.alloc_shared((chunk_size, bdim), dtype)
                x_local = T.alloc_fragment((chunk_size, bdim), dtype)
                output_local = T.alloc_fragment((bdim,), accum_dtype)
                start_token = T.alloc_var(T.int32)
                end_token = T.alloc_var(T.int32)
                if use_offsets == 0:
                    start_token = i_t * chunk_size
                    end_token = T.min(start_token + chunk_size, seq_len)
                else:
                    seq_id = indices[i_t, 0]
                    local_chunk_id = indices[i_t, 1]
                    start_token = offsets[seq_id] + local_chunk_id * chunk_size
                    end_token = T.min(start_token + chunk_size, offsets[seq_id + 1])
                start_dim = i_d * bdim
                end_dim = T.min(start_dim + bdim, dim)
                if use_offsets == 0 and seq_len % chunk_size == 0 and (dim % bdim == 0):
                    T.copy(x[i_b, start_token:start_token + chunk_size, i_h, start_dim:end_dim], x_shared)
                else:
                    T.clear(x_shared)
                    T.copy(x[i_b, start_token:end_token, i_h, start_dim:end_dim], x_shared[0:end_token - start_token, :end_dim - start_dim], disable_tma=True)
                T.copy(x_shared, x_local)
                T.reduce_sum(x_local, output_local, dim=0)
                for d_idx in T.Parallel(bdim):
                    o[i_b, i_t, i_h, start_dim + d_idx] = T.cast(output_local[d_idx] / T.cast(end_token - start_token, accum_dtype), dtype)
        return _mean_pooling_main
    return _mean_pooling_func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
