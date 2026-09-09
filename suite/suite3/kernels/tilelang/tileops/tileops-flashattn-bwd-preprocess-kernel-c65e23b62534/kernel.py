# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

from typing import Callable, Optional, Tuple
import tilelang.language as T
import tilelang

@tilelang.jit(out_idx=[2])
def _kernel_factory(batch: int, heads: int, seq_len: int, dim: int, dtype: str) -> Callable:
    accum_dtype = 'float'
    shape = (batch, seq_len, heads, dim)
    blk = 256

    @T.prim_func
    def flash_bwd_prep(o: T.Tensor(shape, dtype), do: T.Tensor(shape, dtype), delta: T.Tensor([batch, heads, seq_len], accum_dtype)) -> None:
        with T.Kernel(heads, T.ceildiv(seq_len, blk), batch) as (bx, by, bz):
            o_frag = T.alloc_fragment([blk, blk], dtype)
            do_frag = T.alloc_fragment([blk, blk], dtype)
            acc = T.alloc_fragment([blk, blk], accum_dtype)
            delta_frag = T.alloc_fragment([blk], accum_dtype)
            T.clear(acc)
            for k in range(T.ceildiv(dim, blk)):
                T.copy(o[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], o_frag)
                T.copy(do[bz, by * blk:(by + 1) * blk, bx, k * blk:(k + 1) * blk], do_frag)
                for i, j in T.Parallel(blk, blk):
                    acc[i, j] += o_frag[i, j] * do_frag[i, j]
            T.reduce_sum(acc, delta_frag, 1)
            T.copy(delta_frag, delta[bz, bx, by * blk:(by + 1) * blk])
    return flash_bwd_prep

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
