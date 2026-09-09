# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import math
import tilelang
_SCATTER_THREADS = 256

def _kernel_factory(numel: int, num_experts: int, block_size: int):
    """K2: multi-block scatter using global atomicAdd on cumsum buffer.

    cumsum[e] holds the current write offset for expert e (set by K1, then
    incremented here). CUDA stream order guarantees K1 finishes before K2.

    Reference: count_and_sort_expert_tokens_kernel in sgl-kernel.
    """
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    scatter_blocks = min(math.ceil(numel / _SCATTER_THREADS), 65535)
    total_scatter_threads = scatter_blocks * _SCATTER_THREADS

    @tilelang.jit(out_idx=[], compile_flags=['-O3'])
    def _scatter():

        @T.prim_func
        def _scatter_main(flat: T.Tensor([numel], 'int32'), sorted_token_ids: T.Tensor([max_padded], 'int32'), cumsum: T.Tensor([num_experts + 1], 'int32')):
            with T.Kernel(scatter_blocks, threads=_SCATTER_THREADS) as (bid,):
                tx = T.get_thread_binding()
                gid = bid * _SCATTER_THREADS + tx
                slot_buf = T.alloc_local([1], 'int32')
                for i in T.serial(T.ceildiv(numel, total_scatter_threads)):
                    idx = gid + i * total_scatter_threads
                    if idx < numel:
                        eid = flat[idx]
                        slot_buf[0] = T.atomic_add(cumsum[eid], T.int32(1), return_prev=True)
                        slot = slot_buf[0]
                        sorted_token_ids[slot] = idx
        return _scatter_main
    return _scatter

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
