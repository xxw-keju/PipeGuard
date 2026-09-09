# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dim_k: int, dtype: str='float32'):
    """Merge dk contributions and convert chunk-local dg_cum to dg."""
    block_C = chunk_size

    @tilelang.jit(out_idx=[-2, -1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func(threads=128):

        @T.prim_func
        def merge_bwd_outputs(dk_partial: T.Tensor([batch, head, seq_len, dim_k], dtype), dk_corr: T.Tensor([batch, head, seq_len, dim_k], dtype), dk_prepare: T.Tensor([batch, head, seq_len, dim_k], dtype), dg_partial: T.Tensor([batch, head, seq_len], dtype), dg_corr: T.Tensor([batch, head, seq_len], dtype), dg_prepare: T.Tensor([batch, head, seq_len], dtype), dk: T.Tensor([batch, head, seq_len, dim_k], dtype), dg: T.Tensor([batch, head, seq_len], dtype)):
            with T.Kernel(batch, head, seq_len // block_C, threads=threads) as (bid, hid, cid):
                offset = cid * block_C
                for n, kk in T.Parallel(block_C, dim_k):
                    dk[bid, hid, offset + n, kk] = dk_partial[bid, hid, offset + n, kk] + dk_corr[bid, hid, offset + n, kk] + dk_prepare[bid, hid, offset + n, kk]
                for n in T.Parallel(block_C):
                    acc = T.alloc_var(T.float32, init=0.0)
                    for j in T.Serial(block_C):
                        if j >= n:
                            acc += T.cast(dg_partial[bid, hid, offset + j] + dg_corr[bid, hid, offset + j] + dg_prepare[bid, hid, offset + j], 'float32')
                    dg[bid, hid, offset + n] = T.cast(acc, dtype)
        return merge_bwd_outputs
    return _func

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
