# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dtype: str):
    num_chunks = seq_len // chunk_size

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func():

        @T.prim_func
        def chunk_cumsum_bthd_kernel(g: T.Tensor([batch, seq_len, head], dtype), out: T.Tensor([batch, seq_len, head], dtype)):
            with T.Kernel(batch, num_chunks, threads=128) as (bid, cid):
                base = cid * chunk_size
                acc_s = T.alloc_shared([head], 'float32')
                for hid in T.Parallel(head):
                    acc_s[hid] = T.float32(0.0)
                for i in T.Serial(chunk_size):
                    for hid in T.Parallel(head):
                        acc_s[hid] = acc_s[hid] + T.cast(g[bid, base + i, hid], 'float32')
                        out[bid, base + i, hid] = T.cast(acc_s[hid], dtype)
        return chunk_cumsum_bthd_kernel
    return _func()

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
