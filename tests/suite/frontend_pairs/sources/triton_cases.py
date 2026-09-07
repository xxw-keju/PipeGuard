from __future__ import annotations

from pathlib import Path

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl
from triton.experimental.gluon.language.nvidia import ampere, hopper
from triton.experimental.gluon.language.nvidia.blackwell import mbarrier, tma


@gluon.constexpr_function
def _cga_layout():
    return []


@gluon.constexpr_function
def _blocked_1d(threads: int):
    return ttgl.BlockedLayout(
        size_per_thread=[1],
        threads_per_warp=[32],
        warps_per_cta=[threads // 32],
        order=[0],
        cga_layout=_cga_layout(),
    )


def _memory_kernel(family: str, elements: int, threads: int):
    THREADS = ttgl.constexpr(threads)
    ELEMENTS = ttgl.constexpr(elements)
    IS_CTA = ttgl.constexpr(family == "CTA-HANDOFF")
    IS_WAW = ttgl.constexpr(family == "MEM-WAW")

    @gluon.jit
    def kernel(data, out, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = _blocked_1d(THREADS)
        lanes = ttgl.arange(0, THREADS, layout)
        ttgl.store(data + lanes * 0, 1, mask=lanes == 0)
        if IS_CTA and not BUGGY:
            ttgl.barrier()
        index: ttgl.constexpr = 0 if BUGGY or IS_CTA else ELEMENTS
        if IS_WAW:
            ttgl.store(data + lanes * 0 + index, 2, mask=lanes == 1)
        else:
            value = ttgl.load(data + lanes * 0 + index, mask=lanes == 1, other=0)
            ttgl.store(out + lanes, value, mask=lanes == 1)

    return kernel


def _mbarrier_kernel():
    @gluon.jit
    def kernel(BUGGY: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.arrive(bar, count=1)
        if not BUGGY:
            mbarrier.wait(bar, phase=0)
        mbarrier.arrive(bar, count=1)
        if BUGGY:
            mbarrier.wait(bar, phase=0)
        mbarrier.wait(bar, phase=1)

    return kernel


def _cp_async_kernel(family: str, threads: int):
    THREADS = ttgl.constexpr(threads)
    IS_OVERLAP = ttgl.constexpr(family == "CP-OVERLAP")
    IS_WAIT = ttgl.constexpr(family == "CP-WAIT")

    @gluon.jit
    def kernel(source, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = _blocked_1d(THREADS)
        offsets = ttgl.arange(0, THREADS, layout)
        shared = ttgl.allocate_shared_memory(ttgl.int32, [2, THREADS],
                                             ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1,
                                                                      order=[0], cga_layout=_cga_layout()))
        ampere.async_copy.async_load(shared.index(0), source + offsets)
        if IS_OVERLAP:
            destination = shared.index(0 if BUGGY else 1)
            ampere.async_copy.async_load(destination, source + offsets + THREADS)
        ampere.async_copy.commit_group()
        if IS_WAIT and BUGGY:
            shared.index(0).store(ttgl.full([THREADS], 0, ttgl.int32, layout))
        ampere.async_copy.wait_group(0)
        if IS_WAIT and not BUGGY:
            shared.index(0).store(ttgl.full([THREADS], 0, ttgl.int32, layout))

    return kernel


def _tma_load_kernel(family: str):
    IS_PROXY = ttgl.constexpr(family == "TMA-PROXY")
    IS_OBSERVE = ttgl.constexpr(family == "TMA-OBSERVE")

    @gluon.jit
    def kernel(input_desc, input_ptr, output, BUGGY: ttgl.constexpr):
        block: ttgl.constexpr = 32
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(ttgl.float16, [block, block], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        if IS_PROXY:
            offsets = ttgl.arange(0, block, ttgl.SliceLayout(1, layout))
            ttgl.store(input_ptr + offsets, 1.0, mask=offsets == 0)
            if not BUGGY:
                hopper.fence_async_shared()
        mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar, shared)
        if IS_OBSERVE and BUGGY:
            value = shared.load(layout)
            mbarrier.wait(bar, 0, deps=[shared])
        else:
            mbarrier.wait(bar, 0, deps=[shared])
            value = shared.load(layout)
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, layout))[None, :]
        ttgl.store(output + rows * block + cols, value)
        mbarrier.invalidate(bar)

    return kernel


def _wgmma_kernel():
    @gluon.jit
    def kernel(output, BUGGY: ttgl.constexpr, OPERAND_CASE: ttgl.constexpr):
        block: ttgl.constexpr = 128
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2, cga_layout=_cga_layout())
        blocked: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16], cga_layout=_cga_layout())
        a = ttgl.allocate_shared_memory(ttgl.float16, [block, 64], shared_layout)
        b = ttgl.allocate_shared_memory(ttgl.float16, [64, block], shared_layout)
        acc = hopper.warpgroup_mma_init(ttgl.zeros([block, block], ttgl.float32, mma_layout))
        acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
        if OPERAND_CASE:
            if BUGGY:
                a.store(ttgl.zeros([block, 64], ttgl.float16, blocked))
            hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
            if not BUGGY:
                a.store(ttgl.zeros([block, 64], ttgl.float16, blocked))
        else:
            acc = hopper.warpgroup_mma_wait(
                num_outstanding=1 if BUGGY else 0,
                deps=[acc],
            )
            rows = ttgl.arange(0, block)[:, None]
            cols = ttgl.arange(0, block)[None, :]
            ttgl.store(output + rows * block + cols, acc)

    return kernel


def _tma_store_kernel():
    @gluon.jit
    def kernel(output_desc, BUGGY: ttgl.constexpr, SOURCE_CASE: ttgl.constexpr):
        block: ttgl.constexpr = 32
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(ttgl.float16, [block, block], shared_layout)
        tma.async_store(output_desc, [0, 0], shared)
        if SOURCE_CASE:
            if BUGGY:
                shared.store(ttgl.zeros([block, block], ttgl.float16, layout))
            tma.store_wait(pendings=0)
            if not BUGGY:
                shared.store(ttgl.zeros([block, block], ttgl.float16, layout))
        else:
            tma.store_wait(pendings=1)
            if BUGGY:
                tma.async_store(output_desc, [0, 0], shared)
            tma.store_wait(pendings=0)
            if not BUGGY:
                tma.async_store(output_desc, [0, 0], shared)
            tma.store_wait(pendings=0)

    return kernel


def compile_triton_case(family: str, variant: int, polarity: str) -> str:
    buggy = polarity == "BUG"
    elements = (1, 2, 4, 8)[variant % 4]
    threads = 128 if family == "WGMMA-ENDPOINT" else (32, 64)[variant % 2]
    num_warps = max(1, threads // 32)

    if family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        kernel = _memory_kernel(family, elements, threads)
        data = torch.empty((max(16, elements + 1),), device="cuda", dtype=torch.int32)
        out = torch.empty((threads,), device="cuda", dtype=torch.int32)
        compiled = kernel.warmup(data, out, BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family == "MB-REUSE":
        compiled = _mbarrier_kernel().warmup(BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family in {"CP-WAIT", "CP-OVERLAP"}:
        kernel = _cp_async_kernel(family, threads)
        source = torch.empty((2 * threads,), device="cuda", dtype=torch.int32)
        compiled = kernel.warmup(source, BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family in {"TMA-OBSERVE", "TMA-PROXY"}:
        block = 32
        tensor = torch.empty((block, block), device="cuda", dtype=torch.float16)
        output = torch.empty_like(tensor)
        shared_layout = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=())
        descriptor = gluon.nvidia.hopper.TensorDescriptor.from_tensor(tensor, [block, block], shared_layout)
        compiled = _tma_load_kernel(family).warmup(
            descriptor, tensor, output, BUGGY=buggy, grid=(1,), num_warps=4, num_ctas=1)
    elif family == "WGMMA-ENDPOINT":
        output = torch.empty((128, 128), device="cuda", dtype=torch.float32)
        compiled = _wgmma_kernel().warmup(
            output, BUGGY=buggy, OPERAND_CASE=variant % 2 == 0,
            grid=(1,), num_warps=4, num_ctas=1)
    elif family == "TMA-STORE-ENDPOINT":
        block = 32
        output = torch.empty((block, block), device="cuda", dtype=torch.float16)
        shared_layout = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=())
        descriptor = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [block, block], shared_layout)
        compiled = _tma_store_kernel().warmup(
            descriptor, BUGGY=buggy, SOURCE_CASE=variant % 2 == 0,
            grid=(1,), num_warps=4, num_ctas=1)
    else:
        raise KeyError(f"unsupported Triton frontend-pair family: {family}")
    return str(compiled.asm["ttgir"])


def write_triton_case(path: str | Path, family: str, variant: int, polarity: str) -> None:
    Path(path).write_text(compile_triton_case(family, variant, polarity))
