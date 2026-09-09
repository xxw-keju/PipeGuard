# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _kernel_factory(batch: int, head: int, seq_len: int, chunk_size: int, dtype: str):
    num_chunks = seq_len // chunk_size

    @tilelang.jit(out_idx=[-1], pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True}, compile_flags=['-O3', '-DENABLE_BF16'])
    def _func():

        @T.prim_func
        def prefill_chunk_cumsum_bhtd(g: T.Tensor([batch, head, seq_len], dtype), out: T.Tensor([batch, head, seq_len], dtype)):
            with T.Kernel(batch * head * num_chunks, threads=128) as bx:
                bid = bx // (head * num_chunks)
                hid = bx // num_chunks % head
                cid = bx % num_chunks
                base = cid * chunk_size
                g_s = T.alloc_shared([chunk_size], dtype)
                out_s = T.alloc_shared([chunk_size], 'float32')
                T.copy(g[bid, hid, base:base + chunk_size], g_s, disable_tma=True)
                out_s[0] = T.cast(g_s[0], 'float32')
                for i in T.Serial(1, chunk_size):
                    out_s[i] = out_s[i - 1] + T.cast(g_s[i], 'float32')
                for i in T.Parallel(chunk_size):
                    out[bid, hid, base + i] = T.cast(out_s[i], dtype)
        return prefill_chunk_cumsum_bhtd
    return _func()

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
