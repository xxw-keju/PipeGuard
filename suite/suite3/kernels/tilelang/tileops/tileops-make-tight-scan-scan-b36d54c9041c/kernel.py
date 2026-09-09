# Normalized device code for PipeGuard Suite 3.
# Exact upstream code and provenance are stored beside this file.

import tilelang.language as T
import tilelang

def _make_tight_scan_body(numel: int, num_experts: int, top_k: int, threads: int):
    """Shared tight count/prefix/scatter body for split and fused launches."""

    @T.macro
    def scan(flat_ids, physical_ends, permuted_idx, inverse_indices, write_offsets, counts, offsets, slot_buf):
        tx = T.get_thread_binding()
        for i in T.serial(T.ceildiv(num_experts, threads)):
            idx = i * threads + tx
            if idx < num_experts:
                counts[idx] = T.int32(0)
        T.sync_threads()
        for i in T.serial(T.ceildiv(numel, threads)):
            idx = i * threads + tx
            if idx < numel:
                T.atomic_add(counts[flat_ids[idx]], 1)
        T.sync_threads()
        if tx == 0:
            offsets[0] = T.int64(0)
            for expert in T.serial(num_experts):
                offsets[expert + 1] = offsets[expert] + T.Cast(T.int64, counts[expert])
        T.sync_threads()
        for i in T.serial(T.ceildiv(num_experts, threads)):
            idx = i * threads + tx
            if idx < num_experts:
                physical_ends[idx] = T.Cast(T.int32, offsets[idx + 1])
                write_offsets[idx] = T.Cast(T.int32, offsets[idx])
        T.sync_threads()
        for i in T.serial(T.ceildiv(numel, threads)):
            idx = i * threads + tx
            if idx < numel:
                expert = flat_ids[idx]
                slot_buf[0] = T.atomic_add(write_offsets[expert], T.int32(1), return_prev=True)
                slot = slot_buf[0]
                permuted_idx[slot] = idx // T.int32(top_k)
                inverse_indices[idx] = slot
    return scan

def _kernel_factory(numel: int, num_experts: int, top_k: int):
    """Count assignments and produce tight physical-PSUM metadata."""

    @tilelang.jit(out_idx=[], compile_flags=['-O3'])
    def _scan(threads: int):
        scan = _make_tight_scan_body(numel, num_experts, top_k, threads)

        @T.prim_func
        def _scan_main(flat_ids: T.Tensor([numel], 'int32'), physical_ends: T.Tensor([num_experts], 'int32'), permuted_idx: T.Tensor([numel], 'int32'), inverse_indices: T.Tensor([numel], 'int32'), write_offsets: T.Tensor([num_experts], 'int32')):
            with T.Kernel(1, threads=threads) as (_,):
                counts = T.alloc_shared([num_experts], 'int32')
                offsets = T.alloc_shared([num_experts + 1], 'int64')
                slot_buf = T.alloc_local([1], 'int32')
                scan(flat_ids, physical_ends, permuted_idx, inverse_indices, write_offsets, counts, offsets, slot_buf)
        return _scan_main
    return _scan

def build_kernel(case):
    """Construct the TileLang device entry without launching it."""
    return _kernel_factory(**case['build']['factory_arguments'])
