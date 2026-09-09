# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import math
import tilelang
_FILL_THREADS = 256

def _kernel_factory(numel: int, num_experts: int, block_size: int):
    """Single fused kernel for small batches (numel < 1024, num_experts <= 64).

    Uses per-worker private count rows to eliminate atomic conflicts in count
    and scatter phases. Concurrent sentinel fill (fill_threads) + compute
    (worker_threads) within the same block.

    Uses 0-indexed expert IDs throughout (unlike sgl-kernel which uses 1-indexed).
    Output format matches the large-batch path (K1+K2).

    Reference: moe_align_block_size_small_batch_expert_kernel in sgl-kernel.
    """
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)
    worker_threads = max(num_experts, 32)
    total_threads = _FILL_THREADS + worker_threads
    cnts_size = (worker_threads + 1) * num_experts

    @tilelang.jit(out_idx=[], compile_flags=['-O3'])
    def _small():

        @T.prim_func
        def _small_main(flat: T.Tensor([numel], 'int32'), sorted_token_ids: T.Tensor([max_padded], 'int32'), expert_ids: T.Tensor([max_num_blocks], 'int32'), num_tokens_post_pad: T.Tensor([1], 'int32')):
            with T.Kernel(1, threads=total_threads) as (_,):
                tx = T.get_thread_binding()
                cumsum_s = T.alloc_shared([num_experts + 1], 'int32')
                tokens_cnts = T.alloc_shared([cnts_size], 'int32')
                if tx < _FILL_THREADS:
                    for i in T.serial(T.ceildiv(max_padded, _FILL_THREADS)):
                        idx = i * _FILL_THREADS + tx
                        if idx < max_padded:
                            sorted_token_ids[idx] = numel
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS
                    for i in T.serial(num_experts):
                        tokens_cnts[(wid + 1) * num_experts + i] = T.int32(0)
                    for i in T.serial(T.ceildiv(numel, worker_threads)):
                        idx = wid + i * worker_threads
                        if idx < numel:
                            eid = flat[idx]
                            tokens_cnts[(wid + 1) * num_experts + eid] = tokens_cnts[(wid + 1) * num_experts + eid] + 1
                T.sync_threads()
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS
                    if wid < num_experts:
                        tokens_cnts[wid] = T.int32(0)
                        for k in T.serial(worker_threads):
                            tokens_cnts[(k + 1) * num_experts + wid] = tokens_cnts[(k + 1) * num_experts + wid] + tokens_cnts[k * num_experts + wid]
                T.sync_threads()
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS
                    if wid == 0:
                        cumsum_s[0] = T.int32(0)
                        for e in T.serial(num_experts):
                            cnt = tokens_cnts[worker_threads * num_experts + e]
                            cumsum_s[e + 1] = cumsum_s[e] + T.ceildiv(cnt, block_size) * block_size
                        num_tokens_post_pad[0] = cumsum_s[num_experts]
                T.sync_threads()
                if tx >= _FILL_THREADS:
                    wid = tx - _FILL_THREADS
                    if wid < num_experts:
                        e_start = cumsum_s[wid] // block_size
                        e_end = cumsum_s[wid + 1] // block_size
                        for b in T.serial(max_num_blocks):
                            blk = e_start + b
                            if blk < e_end:
                                expert_ids[blk] = wid
                    for i in T.serial(T.ceildiv(numel, worker_threads)):
                        idx = wid + i * worker_threads
                        if idx < numel:
                            eid = flat[idx]
                            rank = tokens_cnts[wid * num_experts + eid]
                            slot = rank + cumsum_s[eid]
                            sorted_token_ids[slot] = idx
                            tokens_cnts[wid * num_experts + eid] = rank + 1
        return _small_main
    return _small

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
