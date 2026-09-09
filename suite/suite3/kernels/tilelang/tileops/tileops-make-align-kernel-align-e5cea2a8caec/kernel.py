# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import math
import tilelang

def _kernel_factory(numel: int, num_experts: int, block_size: int):
    """K1: count → warp-scan → expert_ids → sentinel fill. Does NOT scatter."""
    max_padded = numel + (num_experts + 1) * (block_size - 1)
    max_num_blocks = math.ceil(max_padded / block_size)

    @tilelang.jit(out_idx=[], compile_flags=['-O3'])
    def _align(threads: int):

        @T.prim_func
        def _align_main(flat: T.Tensor([numel], 'int32'), sorted_token_ids: T.Tensor([max_padded], 'int32'), expert_ids: T.Tensor([max_num_blocks], 'int32'), num_tokens_post_pad: T.Tensor([1], 'int32'), cumsum: T.Tensor([num_experts + 1], 'int32')):
            with T.Kernel(1, threads=threads) as (_,):
                tx = T.get_thread_binding()
                num_warps = threads // 32
                s_counts = T.alloc_shared([num_experts], 'int32')
                s_vals = T.alloc_shared([threads], 'int32')
                s_warp_sum = T.alloc_shared([num_warps], 'int32')
                s_warp_excl = T.alloc_shared([num_warps], 'int32')
                s_total = T.alloc_shared([1], 'int32')
                s_cumsum = T.alloc_shared([num_experts + 1], 'int32')
                for i in T.serial(T.ceildiv(num_experts, threads)):
                    idx = i * threads + tx
                    if idx < num_experts:
                        s_counts[idx] = T.int32(0)
                T.sync_threads()
                for i in T.serial(T.ceildiv(numel, threads)):
                    idx = i * threads + tx
                    if idx < numel:
                        T.atomic_add(s_counts[flat[idx]], 1)
                T.sync_threads()
                lane = tx % 32
                warp_id = tx // 32
                s_vals[tx] = T.ceildiv(s_counts[tx], block_size) * block_size if tx < num_experts else T.int32(0)
                T.sync_threads()
                for d in T.serial(5):
                    stride = 1 << d
                    up_val = T.tvm_warp_shuffle_up(T.uint32(4294967295), s_vals[tx], stride, 32, 32)
                    if lane >= stride:
                        s_vals[tx] = s_vals[tx] + up_val
                T.sync_threads()
                if lane == 31:
                    s_warp_sum[warp_id] = s_vals[tx]
                T.sync_threads()
                for w in T.serial(num_warps):
                    if tx == 0:
                        if w == 0:
                            s_total[0] = T.int32(0)
                        s_warp_excl[w] = s_total[0]
                        s_total[0] = s_total[0] + s_warp_sum[w]
                T.sync_threads()
                own_padded = T.ceildiv(s_counts[tx], block_size) * block_size if tx < num_experts else T.int32(0)
                excl = s_vals[tx] - own_padded + s_warp_excl[warp_id]
                if tx < num_experts:
                    s_cumsum[tx] = excl
                    cumsum[tx] = excl
                if tx == num_experts - 1:
                    total = s_vals[tx] + s_warp_excl[warp_id]
                    s_cumsum[num_experts] = total
                    cumsum[num_experts] = total
                    num_tokens_post_pad[0] = total
                T.sync_threads()
                if tx < num_experts:
                    e_start = s_cumsum[tx] // block_size
                    e_end = s_cumsum[tx + 1] // block_size
                    for b in T.serial(max_num_blocks):
                        blk = e_start + b
                        if blk < e_end:
                            expert_ids[blk] = tx
                for i in T.serial(T.ceildiv(max_padded, threads)):
                    idx = i * threads + tx
                    if idx < max_padded:
                        sorted_token_ids[idx] = numel
        return _align_main
    return _align

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
