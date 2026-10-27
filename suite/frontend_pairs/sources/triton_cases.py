from __future__ import annotations

from pathlib import Path

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
    def kernel(BUGGY: ttgl.constexpr, DEPTH: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        for phase in ttgl.static_range(0, DEPTH):
            mbarrier.arrive(bar, count=1)
            mbarrier.wait(bar, phase=phase % 2)
        if BUGGY:
            mbarrier.arrive(bar, count=1)
            mbarrier.arrive(bar, count=1)
            mbarrier.wait(bar, phase=DEPTH % 2)
            mbarrier.wait(bar, phase=(DEPTH + 1) % 2)

    return kernel


def _suite2_diverse_mbarrier_kernel(scenario: str):
    MISSING_ARRIVAL = ttgl.constexpr(scenario == "MBARRIER:MISSING-ARRIVAL")
    NEXT_BEFORE_WAIT = ttgl.constexpr(scenario == "MBARRIER:NEXT-ARRIVE-BEFORE-WAIT")
    WRONG_PARITY = ttgl.constexpr(scenario == "MBARRIER:WRONG-PARITY-WAIT")
    DELAYED_NEXT = ttgl.constexpr(scenario == "MBARRIER:DELAYED-NEXT-ARRIVE-BEFORE-WAIT")
    LATE_MISSING = ttgl.constexpr(scenario == "MBARRIER:LATE-MISSING-ARRIVAL")
    INIT_COUNT = ttgl.constexpr(2 if MISSING_ARRIVAL or LATE_MISSING else 1)

    @gluon.jit
    def kernel(data, out, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = _blocked_1d(32)
        lanes = ttgl.arange(0, 32, layout)
        shared = ttgl.allocate_shared_memory(
            ttgl.int32,
            [32],
            ttgl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=_cga_layout()
            ),
        )
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=INIT_COUNT)
        if MISSING_ARRIVAL:
            mbarrier.arrive(bar, count=1 if BUGGY else 2)
            mbarrier.wait(bar, phase=0)
        elif NEXT_BEFORE_WAIT:
            mbarrier.arrive(bar, count=1)
            if BUGGY:
                mbarrier.arrive(bar, count=1)
                mbarrier.wait(bar, phase=0)
                mbarrier.wait(bar, phase=1)
            else:
                mbarrier.wait(bar, phase=0)
                mbarrier.arrive(bar, count=1)
                mbarrier.wait(bar, phase=1)
        elif WRONG_PARITY:
            mbarrier.arrive(bar, count=1)
            mbarrier.wait(bar, phase=0)
            mbarrier.arrive(bar, count=1)
            mbarrier.wait(bar, phase=1)
            mbarrier.arrive(bar, count=1)
            mbarrier.wait(bar, phase=1 if BUGGY else 0)
        elif DELAYED_NEXT:
            shared.store(ttgl.full([32], 1, ttgl.int32, layout))
            mbarrier.arrive(bar, count=1)
            if not BUGGY:
                mbarrier.wait(bar, phase=0)
            value = shared.load(layout)
            for step in ttgl.static_range(0, 64):
                value = value + step + 1
                shared.store(value)
            mbarrier.arrive(bar, count=1)
            mbarrier.wait(bar, phase=1)
            observed = shared.load(layout)
            ttgl.store(out + lanes, observed, mask=lanes == 1)
        else:
            mbarrier.arrive(bar, count=2)
            mbarrier.wait(bar, phase=0)
            mbarrier.arrive(bar, count=1 if BUGGY else 2)
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


def _tma_load_kernel(family: str, variant: int = 0):
    IS_PROXY = ttgl.constexpr(family == "TMA-PROXY")
    IS_OBSERVE = ttgl.constexpr(family == "TMA-OBSERVE")
    BLOCK = ttgl.constexpr(32)
    TRANSFERS = ttgl.constexpr(variant + 1)

    @gluon.jit
    def kernel(input_desc, input_ptr, output, BUGGY: ttgl.constexpr):
        block: ttgl.constexpr = BLOCK
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(
            ttgl.float16,
            [TRANSFERS, block, block],
            input_desc.layout,
        )
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        if IS_PROXY:
            offsets = ttgl.arange(0, block, ttgl.SliceLayout(1, layout))
            ttgl.store(input_ptr + offsets, 1.0, mask=offsets == 0)
            if not BUGGY:
                hopper.fence_async_shared()
        mbarrier.expect(bar, input_desc.nbytes_per_cta * TRANSFERS)
        for transfer in ttgl.static_range(0, TRANSFERS):
            tma.async_load(
                input_desc,
                [transfer * block, 0],
                bar,
                shared.index(transfer),
            )
        observed = shared.index(TRANSFERS - 1)
        if IS_OBSERVE and BUGGY:
            value = observed.load(layout)
            mbarrier.wait(bar, 0, deps=[shared])
        else:
            mbarrier.wait(bar, 0, deps=[shared])
            value = observed.load(layout)
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, layout))[None, :]
        ttgl.store(output + rows * block + cols, value)
        mbarrier.invalidate(bar)

    return kernel


def _suite2_tma_sync_kernel(variant: int):
    TRANSFERS = ttgl.constexpr(variant + 1)

    @gluon.jit
    def kernel(input_desc, input_ptr, output, BUGGY: ttgl.constexpr):
        block: ttgl.constexpr = 32
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(ttgl.float16, [block, block], input_desc.layout)
        barrier = mbarrier.allocate_mbarrier()
        mbarrier.init(barrier, count=1)
        mbarrier.expect(barrier, input_desc.nbytes_per_cta * TRANSFERS + 16)
        for transfer in ttgl.static_range(0, TRANSFERS):
            tma.async_load(input_desc, [transfer * block, 0], barrier, shared)
        mbarrier.wait(barrier, 0, deps=[shared])
        value = shared.load(layout)
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, layout))[None, :]
        ttgl.store(output + rows * block + cols, value)
        mbarrier.invalidate(barrier)

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
            result = ttgl.convert_layout(acc, blocked)
            rows = ttgl.arange(0, block, ttgl.SliceLayout(1, blocked))[:, None]
            cols = ttgl.arange(0, block, ttgl.SliceLayout(0, blocked))[None, :]
            ttgl.store(output + rows * block + cols, result)

    return kernel


def _suite2_wgmma_kernel(variant: int, result_class: str):
    REPEATS = ttgl.constexpr(variant + 1)
    MODE = ttgl.constexpr({"SAFE": 0, "DATA_HAZARD": 1, "SYNC_HAZARD": 2}[result_class])

    @gluon.jit
    def kernel(
        output,
        BUGGY: ttgl.constexpr,
        OPERAND_CASE: ttgl.constexpr,
    ):
        block: ttgl.constexpr = 64
        k: ttgl.constexpr = 16
        a_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32,
            element_bitwidth=16,
            rank=2,
            cga_layout=_cga_layout(),
        )
        b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=128,
            element_bitwidth=16,
            rank=2,
            cga_layout=_cga_layout(),
        )
        blocked_a: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[32, 1],
            warps_per_cta=[4, 1],
            order=[1, 0],
            cga_layout=_cga_layout(),
        )
        blocked_b: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[32, 1],
            warps_per_cta=[4, 1],
            order=[0, 1],
            cga_layout=_cga_layout(),
        )
        blocked_c: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1],
            threads_per_warp=[32, 1],
            warps_per_cta=[4, 1],
            order=[0, 1],
            cga_layout=_cga_layout(),
        )
        mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(
            version=[3, 0],
            warps_per_cta=[4, 1],
            instr_shape=[16, 32, 16],
            cga_layout=_cga_layout(),
        )
        a = ttgl.allocate_shared_memory(ttgl.float16, [block, k], a_layout)
        b = ttgl.allocate_shared_memory(ttgl.float16, [k, block], b_layout)
        a.store(ttgl.full([block, k], 1.0, ttgl.float16, blocked_a))
        b.store(ttgl.full([k, block], 1.0, ttgl.float16, blocked_b))
        ttgl.barrier()
        acc = hopper.warpgroup_mma_init(
            ttgl.zeros([block, block], ttgl.float32, mma_layout)
        )
        for _phase in ttgl.static_range(0, REPEATS - 1):
            acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
        acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
        if MODE == 1:
            a.store(ttgl.full([block, k], 1.0, ttgl.float16, blocked_a))
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
        elif MODE == 2:
            acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
        else:
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
        result = ttgl.convert_layout(acc, blocked_c)
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, blocked_c))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, blocked_c))[None, :]
        ttgl.store(output + rows * block + cols, result)
        if MODE == 2:
            hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])

    return kernel


def _suite2_proxy_kernel(variant: int, result_class: str):
    if result_class == "SYNC_HAZARD":
        return _suite2_wgmma_kernel(variant, result_class)

    REPEATS = ttgl.constexpr(variant + 1)
    FENCED = ttgl.constexpr(result_class == "SAFE")

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def producer(shared, layout: ttgl.constexpr):
        for phase in ttgl.static_range(0, REPEATS):
            shared.store(ttgl.full([64, 64], phase + 1, ttgl.float16, layout))

    @gluon.jit
    def kernel(output_desc, output, sink, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            [1, 64], [32, 1], [4, 1], [0, 1], cga_layout=_cga_layout()
        )
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            128, 16, rank=2, cga_layout=_cga_layout()
        )
        shared = ttgl.allocate_shared_memory(ttgl.float16, [64, 64], shared_layout)
        ttgl.warp_specialize(
            [(default_partition, ()), (producer, (shared, layout))],
            [4],
            [32],
        )
        if FENCED:
            hopper.fence_async_shared()
        tma.async_store(output_desc, [0, 0], shared)
        tma.store_wait(0)
        rows = ttgl.arange(0, 64, ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, 64, ttgl.SliceLayout(0, layout))[None, :]
        value = ttgl.load(output + rows * 64 + cols)
        ttgl.store(sink + rows * 64 + cols, value)

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



def _named_memory_kernel(scenario: str, threads: int):
    THREADS = ttgl.constexpr(threads)
    IS_PARTIAL_WAR = ttgl.constexpr(scenario == "MEMORY:PARTIAL-WAR")
    IS_ADJACENT = ttgl.constexpr(scenario == "MEMORY:ADJACENT-DISJOINT")
    IS_DANGLING = ttgl.constexpr(scenario == "MEMORY:DANGLING-REGION")
    IS_CONTAINED = ttgl.constexpr(scenario == "MEMORY:CONTAINED-WAW")
    IS_IDENTITY = ttgl.constexpr(scenario == "MEMORY:ALLOCATION-IDENTITY")

    @gluon.jit
    def kernel(data, other, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = _blocked_1d(THREADS)
        lanes = ttgl.arange(0, THREADS, layout)
        if IS_PARTIAL_WAR:
            value = ttgl.load(data + lanes * 0, mask=lanes == 0, other=0)
            ttgl.store(other + lanes, value, mask=lanes == 0)
            ttgl.store(data + lanes * 0, 1, mask=lanes == 1)
        elif IS_ADJACENT:
            ttgl.store(data + lanes * 0, 1, mask=lanes == 0)
            value = ttgl.load(data + lanes * 0 + 16, mask=lanes == 1, other=0)
            ttgl.store(other + lanes, value, mask=lanes == 1)
        elif IS_DANGLING:
            value = ttgl.load(data + lanes * 0 + 64, mask=lanes == 0, other=0)
            ttgl.store(other + lanes, value, mask=lanes == 0)
        elif IS_CONTAINED:
            ttgl.store(data + lanes * 0, 1, mask=lanes == 0)
            ttgl.store(data + lanes * 0 + (0 if BUGGY else 16), 2, mask=lanes == 1)
        elif IS_IDENTITY:
            ttgl.store(data + lanes * 0, 1, mask=lanes == 0)
            target = data if BUGGY else other
            ttgl.store(target + lanes * 0, 2, mask=lanes == 1)

    return kernel


def _named_cta_kernel(scenario: str, threads: int):
    THREADS = ttgl.constexpr(threads)
    NEED_HANDOFF = ttgl.constexpr(scenario in {
        "CTA-BARRIER:CONTROL-ONLY-HANDOFF",
        "CTA-BARRIER:COMPLETE-HANDOFF",
    })
    DUPLICATE = ttgl.constexpr(scenario == "CTA-BARRIER:DUPLICATE-ARRIVAL")

    @gluon.jit
    def kernel(data, out, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = _blocked_1d(THREADS)
        lanes = ttgl.arange(0, THREADS, layout)
        if NEED_HANDOFF:
            ttgl.store(data + lanes * 0, 1, mask=lanes == 0)
            if not BUGGY:
                ttgl.barrier()
            value = ttgl.load(data + lanes * 0, mask=lanes == 1, other=0)
            ttgl.store(out + lanes, value, mask=lanes == 1)
        else:
            ttgl.barrier()
            if DUPLICATE and BUGGY:
                ttgl.barrier()

    return kernel


def _named_mbarrier_kernel(scenario: str):
    USE_BEFORE_INIT = ttgl.constexpr(scenario == "MBARRIER:USE-BEFORE-INIT")
    TX_MISMATCH = ttgl.constexpr(scenario == "MBARRIER:TX-BYTES-MISMATCH")
    NEXT_BEFORE_WAIT = ttgl.constexpr(scenario == "MBARRIER:NEXT-ARRIVE-BEFORE-WAIT")
    INVALIDATE = ttgl.constexpr(scenario == "MBARRIER:INVALIDATE-REQUIRES-OBSERVATION")
    REINIT = ttgl.constexpr(scenario == "MBARRIER:REINITIALIZE-LIFECYCLE")
    PARITY_WRAP = ttgl.constexpr(scenario == "MBARRIER:PARITY-WRAP")
    STALE_WAIT = ttgl.constexpr(scenario == "MBARRIER:STALE-GENERATION-WAIT")
    TWO_GENERATIONS = ttgl.constexpr(scenario in {
        "MBARRIER:TWO-GENERATION-HANDOFF",
        "MBARRIER:NONCOALESCED-POLL",
    })

    @gluon.jit
    def kernel(BUGGY: ttgl.constexpr):
        bar = mbarrier.allocate_mbarrier()
        if not USE_BEFORE_INIT:
            mbarrier.init(bar, count=1)
        if TX_MISMATCH:
            mbarrier.expect(bar, 8 if BUGGY else 16)
            mbarrier.arrive(bar)
            mbarrier.wait(bar, phase=0)
        elif NEXT_BEFORE_WAIT:
            mbarrier.arrive(bar)
            mbarrier.arrive(bar)
            mbarrier.wait(bar, phase=0)
            mbarrier.wait(bar, phase=1)
        elif INVALIDATE:
            mbarrier.arrive(bar)
            if not BUGGY:
                mbarrier.wait(bar, phase=0)
            mbarrier.invalidate(bar)
        elif REINIT:
            if BUGGY:
                mbarrier.init(bar, count=1)
            else:
                mbarrier.arrive(bar)
                mbarrier.wait(bar, phase=0)
                mbarrier.invalidate(bar)
                mbarrier.init(bar, count=1)
        elif PARITY_WRAP or STALE_WAIT:
            mbarrier.arrive(bar)
            mbarrier.wait(bar, phase=0)
            mbarrier.arrive(bar)
            mbarrier.wait(bar, phase=1)
            mbarrier.arrive(bar)
            mbarrier.wait(bar, phase=1 if PARITY_WRAP and BUGGY else 0)
            if STALE_WAIT and BUGGY:
                mbarrier.wait(bar, phase=0)
        else:
            mbarrier.arrive(bar)
            mbarrier.wait(bar, phase=0)
            if TWO_GENERATIONS:
                mbarrier.arrive(bar)
                mbarrier.wait(bar, phase=1)

    return kernel


def _named_cp_async_kernel(scenario: str, threads: int):
    THREADS = ttgl.constexpr(threads)
    WAIT_BEFORE_COMMIT = ttgl.constexpr(scenario == "CP-ASYNC:WAIT-BEFORE-COMMIT")
    TWO_GROUPS = ttgl.constexpr(scenario == "CP-ASYNC:TWO-GROUP-WAIT-ONE")

    @gluon.jit
    def kernel(source, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = _blocked_1d(THREADS)
        offsets = ttgl.arange(0, THREADS, layout)
        shared = ttgl.allocate_shared_memory(
            ttgl.int32,
            [2, THREADS],
            ttgl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=_cga_layout()
            ),
        )
        ampere.async_copy.async_load(shared.index(0), source + offsets)
        if WAIT_BEFORE_COMMIT:
            ampere.async_copy.wait_group(0)
            shared.index(0).store(ttgl.full([THREADS], 0, ttgl.int32, layout))
        elif TWO_GROUPS:
            ampere.async_copy.commit_group()
            ampere.async_copy.async_load(shared.index(1), source + offsets + THREADS)
            ampere.async_copy.commit_group()
            ampere.async_copy.wait_group(1)
            shared.index(1 if BUGGY else 0).store(ttgl.full([THREADS], 0, ttgl.int32, layout))
            ampere.async_copy.wait_group(0)
        else:
            ampere.async_copy.commit_group()
            ampere.async_copy.wait_group(0)
            shared.index(0).store(ttgl.full([THREADS], 0, ttgl.int32, layout))

    return kernel


def _named_tma_load_kernel(scenario: str):
    IS_PROXY = ttgl.constexpr(scenario in {
        "TMA-LOAD:GENERIC-SOURCE-NO-PROXY",
        "TMA-LOAD:PROXY-AND-WAIT",
    })
    DO_PROXY = ttgl.constexpr(scenario == "TMA-LOAD:PROXY-AND-WAIT")
    EARLY_READ = ttgl.constexpr(scenario == "TMA-LOAD:DESTINATION-BEFORE-WAIT")
    BAD_BYTES = ttgl.constexpr(scenario in {
        "TMA-LOAD:GROUPED-BYTES-MISMATCH",
        "TMA-LOAD:MULTI-TRANSFER-ONE-COMPLETION",
    })

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
            if DO_PROXY:
                hopper.fence_async_shared()
        if BAD_BYTES and BUGGY:
            mbarrier.expect(bar, 16)
        else:
            mbarrier.expect(bar, input_desc.nbytes_per_cta)
        tma.async_load(input_desc, [0, 0], bar, shared)
        if EARLY_READ and BUGGY:
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


def _named_wgmma_kernel(scenario: str):
    TWO_GROUPS = ttgl.constexpr(scenario == "WGMMA:TWO-GROUP-WAIT-ONE")
    DUAL = ttgl.constexpr(scenario == "WGMMA:DUAL-ENDPOINT")

    @gluon.jit
    def kernel(output, BUGGY: ttgl.constexpr):
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
        acc0 = hopper.warpgroup_mma_init(ttgl.zeros([block, block], ttgl.float32, mma_layout))
        acc0 = hopper.warpgroup_mma(a, b, acc0, is_async=True)
        if TWO_GROUPS:
            acc1 = hopper.warpgroup_mma_init(ttgl.zeros([block, block], ttgl.float32, mma_layout))
            acc1 = hopper.warpgroup_mma(a, b, acc1, is_async=True)
            acc1 = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc1])
            chosen = acc1 if BUGGY else acc0
            chosen = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[chosen])
        elif DUAL:
            hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc0])
            a.store(ttgl.zeros([block, 64], ttgl.float16, blocked))
            chosen = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc0])
        else:
            if not BUGGY:
                hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc0])
            a.store(ttgl.zeros([block, 64], ttgl.float16, blocked))
            chosen = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc0])
        result = ttgl.convert_layout(chosen, blocked)
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, blocked))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, blocked))[None, :]
        ttgl.store(output + rows * block + cols, result)

    return kernel


def _named_tma_store_kernel(scenario: str):
    TWO_GROUPS = ttgl.constexpr(scenario == "TMA-STORE:TWO-GROUP-WAIT-ONE")
    WAIT_BEFORE_COMMIT = ttgl.constexpr(scenario == "TMA-STORE:STORE-WAIT-BEFORE-COMMIT")
    INVALID_SOURCE = ttgl.constexpr(scenario == "TMA-STORE:STORE-INVALID-SOURCE")

    @gluon.jit
    def kernel(output_desc, output_ptr, sink, BUGGY: ttgl.constexpr):
        block: ttgl.constexpr = 32
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(ttgl.float16, [block, block], shared_layout)
        if INVALID_SOURCE:
            tma.async_store(output_desc, [0, 0], output_ptr)
        elif WAIT_BEFORE_COMMIT:
            tma.store_wait(pendings=0)
            tma.async_store(output_desc, [0, 0], shared)
            shared.store(ttgl.zeros([block, block], ttgl.float16, layout))
        elif TWO_GROUPS:
            tma.async_store(output_desc, [0, 0], shared)
            tma.async_store(output_desc, [block, 0], shared)
            tma.store_wait(pendings=1)
            value = ttgl.load(output_ptr + (block * block if BUGGY else 0))
            ttgl.store(sink, value)
            tma.store_wait(pendings=0)
        else:
            tma.async_store(output_desc, [0, 0], shared)
            tma.store_wait(pendings=0)
            shared.store(ttgl.zeros([block, block], ttgl.float16, layout))

    return kernel

def compile_triton_case(family: str, scenario_id: str, variant: int, polarity: str) -> str:
    import torch  # lazy: compile/warmup paths only; keeps module importable without torch

    buggy = polarity == "BUG"
    elements = (1, 2, 4, 8)[variant % 4]
    threads = 128 if family == "WGMMA-ENDPOINT" else (32, 64)[variant % 2]
    num_warps = max(1, threads // 32)

    if family == "MEMORY":
        kernel = _named_memory_kernel(scenario_id, threads)
        data = torch.empty((128,), device="cuda", dtype=torch.int32)
        other = torch.empty((128,), device="cuda", dtype=torch.int32)
        compiled = kernel.warmup(data, other, BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family == "CTA-BARRIER":
        kernel = _named_cta_kernel(scenario_id, threads)
        data = torch.empty((128,), device="cuda", dtype=torch.int32)
        out = torch.empty((128,), device="cuda", dtype=torch.int32)
        compiled = kernel.warmup(data, out, BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family == "MBARRIER":
        compiled = _named_mbarrier_kernel(scenario_id).warmup(
            BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family == "CP-ASYNC":
        source = torch.empty((2 * threads,), device="cuda", dtype=torch.int32)
        compiled = _named_cp_async_kernel(scenario_id, threads).warmup(
            source, BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family == "TMA-LOAD":
        block = 32
        tensor = torch.empty((block, block), device="cuda", dtype=torch.float16)
        output = torch.empty_like(tensor)
        shared_layout = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=())
        descriptor = gluon.nvidia.hopper.TensorDescriptor.from_tensor(tensor, [block, block], shared_layout)
        compiled = _named_tma_load_kernel(scenario_id).warmup(
            descriptor, tensor, output, BUGGY=buggy, grid=(1,), num_warps=4, num_ctas=1)
    elif family == "WGMMA":
        output = torch.empty((128, 128), device="cuda", dtype=torch.float32)
        named_num_warps = 2 if scenario_id == "WGMMA:INVALID-WARPGROUP-SHAPE" else 4
        compiled = _named_wgmma_kernel(scenario_id).warmup(
            output, BUGGY=buggy, grid=(1,), num_warps=named_num_warps, num_ctas=1)
    elif family == "TMA-STORE":
        block = 32
        output = torch.empty((2 * block, block), device="cuda", dtype=torch.float16)
        sink = torch.empty((1,), device="cuda", dtype=torch.float16)
        shared_layout = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=())
        descriptor = gluon.nvidia.hopper.TensorDescriptor.from_tensor(output, [block, block], shared_layout)
        compiled = _named_tma_store_kernel(scenario_id).warmup(
            descriptor, output, sink, BUGGY=buggy, grid=(1,), num_warps=4, num_ctas=1)
    elif family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        kernel = _memory_kernel(family, elements, threads)
        data = torch.empty((max(16, elements + 1),), device="cuda", dtype=torch.int32)
        out = torch.empty((threads,), device="cuda", dtype=torch.int32)
        compiled = kernel.warmup(data, out, BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family == "MB-REUSE":
        compiled = _mbarrier_kernel().warmup(
            BUGGY=buggy, DEPTH=variant + 2,
            grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family in {"CP-WAIT", "CP-OVERLAP"}:
        kernel = _cp_async_kernel(family, threads)
        source = torch.empty((2 * threads,), device="cuda", dtype=torch.int32)
        compiled = kernel.warmup(source, BUGGY=buggy, grid=(1,), num_warps=num_warps, num_ctas=1)
    elif family in {"TMA-OBSERVE", "TMA-PROXY"}:
        block = 32
        transfers = variant + 1
        tensor = torch.empty((transfers * block, block), device="cuda", dtype=torch.float16)
        output = torch.empty((block, block), device="cuda", dtype=torch.float16)
        shared_layout = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=())
        descriptor = gluon.nvidia.hopper.TensorDescriptor.from_tensor(tensor, [block, block], shared_layout)
        compiled = _tma_load_kernel(family, variant).warmup(
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


def write_triton_case(
    path: str | Path,
    family: str,
    scenario_id: str,
    variant: int,
    polarity: str,
) -> None:
    Path(path).write_text(compile_triton_case(
        family, scenario_id, variant, polarity
    ))


def _suite2_cp_async_kernel(variant: int, result_class: str):
    GROUPS = ttgl.constexpr(9 + variant)
    THREADS = ttgl.constexpr(32)
    DATA_HAZARD = ttgl.constexpr(result_class == "DATA_HAZARD")
    SYNC_HAZARD = ttgl.constexpr(result_class == "SYNC_HAZARD")
    VALUE = ttgl.constexpr(variant + 1)

    @gluon.jit
    def kernel(source, output):
        layout: ttgl.constexpr = _blocked_1d(THREADS)
        offsets = ttgl.arange(0, THREADS, layout)
        shared = ttgl.allocate_shared_memory(
            ttgl.int32,
            [16, THREADS],
            ttgl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=_cga_layout()
            ),
        )
        for group in ttgl.static_range(0, 8):
            ampere.async_copy.async_load(
                shared.index(group), source + offsets + group * THREADS
            )
            ampere.async_copy.commit_group()
        if not SYNC_HAZARD:
            ampere.async_copy.wait_group(0)
        for group in ttgl.static_range(8, GROUPS):
            ampere.async_copy.async_load(
                shared.index(group), source + offsets + group * THREADS
            )
            ampere.async_copy.commit_group()
        if DATA_HAZARD:
            shared.index(GROUPS - 1).store(
                ttgl.full([THREADS], VALUE, ttgl.int32, layout)
            )
        ampere.async_copy.wait_group(0)
        if not DATA_HAZARD and not SYNC_HAZARD:
            shared.index(GROUPS - 1).store(
                ttgl.full([THREADS], VALUE, ttgl.int32, layout)
            )
        observed = shared.index(GROUPS - 1).load(layout)
        ttgl.store(output + offsets, observed)

    return kernel

# ---------------------------------------------------------------------------
# Suite 2 structural-diversity redesign (2026-10): per-cell variant factories.
#
# Each (primitive, result_class) cell exposes five structurally DISTINCT
# protocol-event-sequence skeletons instead of one skeleton with varying
# repeat counts/shapes.  Factories keep the exact kernel signatures that
# tools/suite2_selected_worker.py triton_prepare() constructs arguments for.
# ---------------------------------------------------------------------------


def _s2v2_cp_async(variant: int, result_class: str):
    V = ttgl.constexpr(variant)
    SAFE = ttgl.constexpr(result_class == "SAFE")
    DH = ttgl.constexpr(result_class == "DATA_HAZARD")
    SH = ttgl.constexpr(result_class == "SYNC_HAZARD")

    @gluon.jit
    def kernel(source, output):
        layout: ttgl.constexpr = _blocked_1d(32)
        offsets = ttgl.arange(0, 32, layout)
        shared = ttgl.allocate_shared_memory(
            ttgl.int32,
            [2, 32],
            ttgl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=_cga_layout()
            ),
        )
        if SAFE:
            if V == 0:
                # single group round trip: issue -> commit -> wait(0) -> read
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(0)
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            elif V == 1:
                # staged waits: wait(1) releases oldest group only; newest
                # buffer is touched only after wait(0)
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.async_load(shared.index(1), source + offsets + 32)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(1)
                first = shared.index(0).load(layout)
                ttgl.store(output + offsets, first)
                ampere.async_copy.wait_group(0)
                second = shared.index(1).load(layout)
                ttgl.store(output + offsets, second)
            elif V == 2:
                # one commit group carrying two copies; wait(0) completes both
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.async_load(shared.index(1), source + offsets + 32)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(0)
                first = shared.index(0).load(layout)
                second = shared.index(1).load(layout)
                ttgl.store(output + offsets, first + second)
            elif V == 3:
                # destination reuse strictly after group completion
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(0)
                shared.index(0).store(ttgl.full([32], 5, ttgl.int32, layout))
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            else:
                # lookahead-1 pipeline: interleaved reads at matching depths
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.async_load(shared.index(1), source + offsets + 32)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(1)
                first = shared.index(0).load(layout)
                ampere.async_copy.wait_group(0)
                second = shared.index(1).load(layout)
                ttgl.store(output + offsets, first + second)
        elif DH:
            if V == 0:
                # missing wait entirely
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            elif V == 1:
                # lax wait: wait(1) tolerates the only group in flight
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(1)
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            elif V == 2:
                # generic overwrite of the in-flight destination before wait
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                shared.index(0).store(ttgl.full([32], 7, ttgl.int32, layout))
                ampere.async_copy.wait_group(0)
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            elif V == 3:
                # overlapping async destinations inside one group (async WAW)
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.async_load(shared.index(0), source + offsets + 32)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(0)
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            else:
                # wrong-buffer read under a staged wait: wait(1) completes only
                # the oldest group but the newest destination is read
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.async_load(shared.index(1), source + offsets + 32)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(1)
                value = shared.index(1).load(layout)
                ttgl.store(output + offsets, value)
        else:
            if V == 0:
                # wait precedes commit: the issued copy belongs to no group yet
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.wait_group(0)
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            elif V == 1:
                # second copy never committed; wait(0) completes only group 1
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.async_load(shared.index(1), source + offsets + 32)
                ampere.async_copy.wait_group(0)
                value = shared.index(1).load(layout)
                ttgl.store(output + offsets, value)
            elif V == 2:
                # wait interposed between issue and its commit: the group is
                # sealed only after the completion point has already passed
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.wait_group(0)
                ampere.async_copy.commit_group()
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)
            elif V == 3:
                # correct first round, wait-before-commit on the second round
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.commit_group()
                ampere.async_copy.wait_group(0)
                first = shared.index(0).load(layout)
                ampere.async_copy.async_load(shared.index(1), source + offsets + 32)
                ampere.async_copy.wait_group(0)
                second = shared.index(1).load(layout)
                ttgl.store(output + offsets, first + second)
            else:
                # bare wait against an empty group stream before any issue
                ampere.async_copy.wait_group(0)
                ampere.async_copy.async_load(shared.index(0), source + offsets)
                ampere.async_copy.wait_group(0)
                value = shared.index(0).load(layout)
                ttgl.store(output + offsets, value)

    return kernel


def _s2v2_tma(variant: int, result_class: str):
    V = ttgl.constexpr(variant)
    SAFE = ttgl.constexpr(result_class == "SAFE")
    DH = ttgl.constexpr(result_class == "DATA_HAZARD")

    @gluon.jit
    def kernel(input_desc, input_ptr, output, BUGGY: ttgl.constexpr):
        block: ttgl.constexpr = 32
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(
            ttgl.float16, [2, block, block], input_desc.layout)
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, layout))[None, :]
        if SAFE:
            if V == 0:
                # canonical single transfer: expect -> issue -> wait -> read
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar, 0, deps=[shared])
                value = shared.index(0).load(layout)
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
            elif V == 1:
                # one expect covering two transfers; single completion
                mbarrier.expect(bar, input_desc.nbytes_per_cta * 2)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                tma.async_load(input_desc, [block, 0], bar, shared.index(1))
                mbarrier.wait(bar, 0, deps=[shared])
                first = shared.index(0).load(layout)
                second = shared.index(1).load(layout)
                ttgl.store(output + rows * block + cols, first + second)
                mbarrier.invalidate(bar)
            elif V == 2:
                # two rounds on one barrier with correct parity flip 0 -> 1
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar, 0, deps=[shared])
                first = shared.index(0).load(layout)
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [block, 0], bar, shared.index(1))
                mbarrier.wait(bar, 1, deps=[shared])
                second = shared.index(1).load(layout)
                ttgl.store(output + rows * block + cols, first + second)
                mbarrier.invalidate(bar)
            elif V == 3:
                # generic global prologue published to the async proxy by an
                # explicit fence before the transfer
                prologue = ttgl.arange(0, block, ttgl.SliceLayout(1, layout))
                ttgl.store(input_ptr + prologue, 1.0, mask=prologue == 0)
                hopper.fence_async_shared()
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar, 0, deps=[shared])
                value = shared.index(0).load(layout)
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
            else:
                # per-slot barriers: each transfer completes on its own barrier
                bar2 = mbarrier.allocate_mbarrier()
                mbarrier.init(bar2, count=1)
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                mbarrier.expect(bar2, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                tma.async_load(input_desc, [block, 0], bar2, shared.index(1))
                mbarrier.wait(bar, 0, deps=[shared])
                mbarrier.wait(bar2, 0, deps=[shared])
                first = shared.index(0).load(layout)
                second = shared.index(1).load(layout)
                ttgl.store(output + rows * block + cols, first + second)
                mbarrier.invalidate(bar)
                mbarrier.invalidate(bar2)
        elif DH:
            if V == 0:
                # destination observed before the completion wait
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                value = shared.index(0).load(layout)
                mbarrier.wait(bar, 0, deps=[shared])
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
            elif V == 1:
                # completion wait missing entirely
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                value = shared.index(0).load(layout)
                ttgl.store(output + rows * block + cols, value)
            elif V == 2:
                # read interposed between two issues of one transaction window
                mbarrier.expect(bar, input_desc.nbytes_per_cta * 2)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                value = shared.index(0).load(layout)
                tma.async_load(input_desc, [block, 0], bar, shared.index(1))
                mbarrier.wait(bar, 0, deps=[shared])
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
            elif V == 3:
                # generic write lands in the in-flight TMA destination
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                shared.index(0).store(ttgl.full([block, block], 0.0, ttgl.float16, layout))
                mbarrier.wait(bar, 0, deps=[shared])
                value = shared.index(0).load(layout)
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
            else:
                # correct first round; second round forgets its wait
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar, 0, deps=[shared])
                first = shared.index(0).load(layout)
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [block, 0], bar, shared.index(1))
                second = shared.index(1).load(layout)
                ttgl.store(output + rows * block + cols, first + second)
        else:
            if V == 0:
                # over-expect: transaction count never reaches zero
                mbarrier.expect(bar, input_desc.nbytes_per_cta + 16)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar, 0, deps=[shared])
                value = shared.index(0).load(layout)
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
            elif V == 1:
                # under-expect: two transfers deliver more bytes than declared
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                tma.async_load(input_desc, [block, 0], bar, shared.index(1))
                mbarrier.wait(bar, 0, deps=[shared])
                first = shared.index(0).load(layout)
                second = shared.index(1).load(layout)
                ttgl.store(output + rows * block + cols, first + second)
                mbarrier.invalidate(bar)
            elif V == 2:
                # stale parity: second round waits on phase 0 again
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar, 0, deps=[shared])
                first = shared.index(0).load(layout)
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [block, 0], bar, shared.index(1))
                mbarrier.wait(bar, 0, deps=[shared])
                second = shared.index(1).load(layout)
                ttgl.store(output + rows * block + cols, first + second)
                mbarrier.invalidate(bar)
            elif V == 3:
                # double-registered expectation: two expects accumulate twice
                # the bytes a single transfer can ever deliver
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar, 0, deps=[shared])
                value = shared.index(0).load(layout)
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
            else:
                # wait targets the wrong barrier object
                bar2 = mbarrier.allocate_mbarrier()
                mbarrier.init(bar2, count=1)
                mbarrier.expect(bar, input_desc.nbytes_per_cta)
                tma.async_load(input_desc, [0, 0], bar, shared.index(0))
                mbarrier.wait(bar2, 0, deps=[shared])
                value = shared.index(0).load(layout)
                ttgl.store(output + rows * block + cols, value)
                mbarrier.invalidate(bar)
                mbarrier.invalidate(bar2)

    return kernel


def _s2v2_mbarrier_data_hazard(variant: int):
    V = ttgl.constexpr(variant)

    @gluon.jit
    def _producer(shared, bar, bar2, layout: ttgl.constexpr, V: ttgl.constexpr):
        if V == 0:
            # arrive precedes the publish write it is meant to order
            mbarrier.arrive(bar, count=1)
            shared.index(0).store(ttgl.full([32, 32], 1.0, ttgl.float16, layout))
        elif V == 1:
            # correct publish+signal; the consumer forgets to wait
            shared.index(0).store(ttgl.full([32, 32], 2.0, ttgl.float16, layout))
            mbarrier.arrive(bar, count=1)
        elif V == 2:
            # correct publish, but the signal goes to a barrier nobody waits on
            shared.index(0).store(ttgl.full([32, 32], 3.0, ttgl.float16, layout))
            mbarrier.arrive(bar, count=1)
        elif V == 3:
            # two generations, correctly re-armed through a reverse handshake
            shared.index(0).store(ttgl.full([32, 32], 4.0, ttgl.float16, layout))
            mbarrier.arrive(bar, count=1)
            mbarrier.wait(bar2, 0)
            shared.index(1).store(ttgl.full([32, 32], 5.0, ttgl.float16, layout))
            mbarrier.arrive(bar, count=1)
        else:
            # slot 1 is written after the only arrive: never published
            shared.index(0).store(ttgl.full([32, 32], 6.0, ttgl.float16, layout))
            mbarrier.arrive(bar, count=1)
            shared.index(1).store(ttgl.full([32, 32], 7.0, ttgl.float16, layout))

    @gluon.jit
    def _consumer(shared, bar, bar2, output, layout: ttgl.constexpr, V: ttgl.constexpr):
        rows = ttgl.arange(0, 32, ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, 32, ttgl.SliceLayout(0, layout))[None, :]
        if V == 0:
            mbarrier.wait(bar, 0, deps=[shared])
            value = shared.index(0).load(layout)
            ttgl.store(output + rows * 32 + cols, value)
        elif V == 1:
            value = shared.index(0).load(layout)
            ttgl.store(output + rows * 32 + cols, value)
        elif V == 2:
            # self-released wrong barrier: orders nothing against the producer
            mbarrier.arrive(bar2, count=1)
            mbarrier.wait(bar2, 0, deps=[shared])
            value = shared.index(0).load(layout)
            ttgl.store(output + rows * 32 + cols, value)
        elif V == 3:
            # generation skew: the gen-2 read happens before its wait
            mbarrier.wait(bar, 0, deps=[shared])
            first = shared.index(0).load(layout)
            mbarrier.arrive(bar2, count=1)
            second = shared.index(1).load(layout)
            mbarrier.wait(bar, 1, deps=[shared])
            ttgl.store(output + rows * 32 + cols, first + second)
        else:
            # slot 1 was never covered by the handshake
            mbarrier.wait(bar, 0, deps=[shared])
            value = shared.index(1).load(layout)
            ttgl.store(output + rows * 32 + cols, value)

    @gluon.jit
    def kernel(input_desc, input_ptr, output, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(
            ttgl.float16,
            [2, 32, 32],
            ttgl.SwizzledSharedLayout(
                vec=1, per_phase=1, max_phase=1, order=[0, 1], cga_layout=_cga_layout()
            ),
        )
        bar = mbarrier.allocate_mbarrier()
        bar2 = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.init(bar2, count=1)
        ttgl.warp_specialize(
            [
                (_consumer, (shared, bar, bar2, output, layout, V)),
                (_producer, (shared, bar, bar2, layout, V)),
            ],
            [4],
            [32],
        )

    return kernel


def _s2v2_wgmma(variant: int, result_class: str):
    V = ttgl.constexpr(variant)
    SAFE = ttgl.constexpr(result_class == "SAFE")
    DH = ttgl.constexpr(result_class == "DATA_HAZARD")

    @gluon.jit
    def kernel(output, BUGGY: ttgl.constexpr, OPERAND_CASE: ttgl.constexpr):
        block: ttgl.constexpr = 64
        k: ttgl.constexpr = 16
        a_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=_cga_layout())
        b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2, cga_layout=_cga_layout())
        blocked_a: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[1, 0], cga_layout=_cga_layout())
        blocked_b: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        blocked_c: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16],
            cga_layout=_cga_layout())
        a = ttgl.allocate_shared_memory(ttgl.float16, [block, k], a_layout)
        b = ttgl.allocate_shared_memory(ttgl.float16, [k, block], b_layout)
        a.store(ttgl.full([block, k], 1.0, ttgl.float16, blocked_a))
        b.store(ttgl.full([k, block], 1.0, ttgl.float16, blocked_b))
        ttgl.barrier()
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, blocked_c))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, blocked_c))[None, :]
        acc = hopper.warpgroup_mma_init(
            ttgl.zeros([block, block], ttgl.float32, mma_layout))
        if SAFE:
            if V == 0:
                # canonical: one async group, wait(0), consume
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            elif V == 1:
                # two independent groups; each result consumed after wait(0)
                acc2 = hopper.warpgroup_mma_init(
                    ttgl.full([block, block], 1.0, ttgl.float32, mma_layout))
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc2 = hopper.warpgroup_mma(a, b, acc2, is_async=True)
                acc2 = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc2])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc2, blocked_c))
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            elif V == 2:
                # depth-1 pipeline: wait(1) between issues, wait(0) at the end
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            elif V == 3:
                # operand buffer reused strictly after wait(0)
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                a.store(ttgl.full([block, k], 2.0, ttgl.float16, blocked_a))
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            else:
                # dual endpoint: redundant second wait(0) on the same group
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
        elif DH:
            if V == 0:
                # shared operand overwritten while the group is in flight
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                a.store(ttgl.full([block, k], 2.0, ttgl.float16, blocked_a))
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            elif V == 1:
                # operand of the *pending* newest group overwritten after wait(1)
                acc2 = hopper.warpgroup_mma_init(
                    ttgl.full([block, block], 1.0, ttgl.float32, mma_layout))
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc2 = hopper.warpgroup_mma(a, b, acc2, is_async=True)
                acc2 = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc2])
                a.store(ttgl.full([block, k], 3.0, ttgl.float16, blocked_a))
                acc2 = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc2])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc2, blocked_c))
            elif V == 2:
                # a CTA barrier is interposed before the overwrite, but a
                # barrier does not order the asynchronous wgmma engine
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                ttgl.barrier()
                a.store(ttgl.full([block, k], 4.0, ttgl.float16, blocked_a))
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            elif V == 3:
                # both operand buffers clobbered while the group is in flight
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                a.store(ttgl.full([block, k], 5.0, ttgl.float16, blocked_a))
                b.store(ttgl.full([k, block], 5.0, ttgl.float16, blocked_b))
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            else:
                # depth-3 chained pipeline: wait(2) still leaves two groups
                # reading the operand that gets rotated underneath them
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=2, deps=[acc])
                a.store(ttgl.full([block, k], 6.0, ttgl.float16, blocked_a))
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
        else:
            if V == 0:
                # lax wait(1): the only group may be incomplete when consumed
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
                hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
            elif V == 1:
                # chained two-group pipeline: wait(1) completes only the
                # oldest group, yet the chained result is consumed
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            elif V == 2:
                # two independent groups: wait(1) can complete only the oldest,
                # yet the newest result is consumed; the oldest is then
                # consumed correctly so both groups stay live
                acc2 = hopper.warpgroup_mma_init(
                    ttgl.full([block, block], 1.0, ttgl.float32, mma_layout))
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc2 = hopper.warpgroup_mma(a, b, acc2, is_async=True)
                acc2 = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc2])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc2, blocked_c))
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            elif V == 3:
                # threshold too lax: wait(2) with a single group outstanding
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=2, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            else:
                # correct round followed by a consume-under-lax-wait round
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
                acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
                acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
                ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))

    return kernel


def _s2v2_proxy(variant: int, result_class: str):
    if result_class == "SYNC_HAZARD":
        return _s2v2_proxy_sync(variant)

    V = ttgl.constexpr(variant)
    FENCED = ttgl.constexpr(result_class == "SAFE")

    @gluon.jit
    def default_partition():
        pass

    @gluon.jit
    def producer(shared, layout: ttgl.constexpr):
        shared.store(ttgl.full([64, 64], 1.0, ttgl.float16, layout))

    @gluon.jit
    def kernel(output_desc, output, sink, BUGGY: ttgl.constexpr):
        layout: ttgl.constexpr = ttgl.BlockedLayout(
            [1, 64], [32, 1], [4, 1], [0, 1], cga_layout=_cga_layout())
        shared_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            128, 16, rank=2, cga_layout=_cga_layout())
        shared = ttgl.allocate_shared_memory(ttgl.float16, [2, 64, 64], shared_layout)
        rows = ttgl.arange(0, 64, ttgl.SliceLayout(1, layout))[:, None]
        cols = ttgl.arange(0, 64, ttgl.SliceLayout(0, layout))[None, :]
        if V == 0:
            # warp-specialized producer publishes across the proxy boundary
            ttgl.warp_specialize(
                [(default_partition, ()), (producer, (shared.index(0), layout))],
                [4],
                [32],
            )
            if FENCED:
                hopper.fence_async_shared()
            tma.async_store(output_desc, [0, 0], shared.index(0))
            tma.store_wait(0)
            value = ttgl.load(output + rows * 64 + cols)
            ttgl.store(sink + rows * 64 + cols, value)
        elif V == 1:
            # minimal publish/store endpoint: no readback consumer
            shared.index(0).store(ttgl.full([64, 64], 2.0, ttgl.float16, layout))
            if FENCED:
                hopper.fence_async_shared()
            tma.async_store(output_desc, [0, 0], shared.index(0))
            tma.store_wait(0)
        elif V == 2:
            # two publish/store rounds through alternating tiles
            shared.index(0).store(ttgl.full([64, 64], 4.0, ttgl.float16, layout))
            if FENCED:
                hopper.fence_async_shared()
            tma.async_store(output_desc, [0, 0], shared.index(0))
            tma.store_wait(0)
            shared.index(1).store(ttgl.full([64, 64], 5.0, ttgl.float16, layout))
            if FENCED:
                hopper.fence_async_shared()
            tma.async_store(output_desc, [0, 0], shared.index(1))
            tma.store_wait(0)
            value = ttgl.load(output + rows * 64 + cols)
            ttgl.store(sink + rows * 64 + cols, value)
        elif V == 3:
            # masked producer handed over by a CTA barrier; the proxy fence is
            # still required because a barrier does not cross proxies
            shared.index(0).store(ttgl.full([64, 64], 6.0, ttgl.float16, layout))
            ttgl.barrier()
            if FENCED:
                hopper.fence_async_shared()
            tma.async_store(output_desc, [0, 0], shared.index(0))
            tma.store_wait(0)
            value = ttgl.load(output + rows * 64 + cols)
            ttgl.store(sink + rows * 64 + cols, value)
        else:
            # staged completion thresholds (SAFE) vs consuming the TMA-written
            # global tile before its completion wait (BUG)
            shared.index(0).store(ttgl.full([64, 64], 8.0, ttgl.float16, layout))
            if FENCED:
                hopper.fence_async_shared()
                tma.async_store(output_desc, [0, 0], shared.index(0))
                tma.store_wait(1)
                tma.store_wait(0)
                value = ttgl.load(output + rows * 64 + cols)
                ttgl.store(sink + rows * 64 + cols, value)
            else:
                tma.async_store(output_desc, [0, 0], shared.index(0))
                value = ttgl.load(output + rows * 64 + cols)
                tma.store_wait(0)
                ttgl.store(sink + rows * 64 + cols, value)

    return kernel


def _s2v2_proxy_sync(variant: int):
    V = ttgl.constexpr(variant)

    @gluon.jit
    def kernel(output, BUGGY: ttgl.constexpr, OPERAND_CASE: ttgl.constexpr):
        block: ttgl.constexpr = 64
        k: ttgl.constexpr = 16
        a_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=_cga_layout())
        b_layout: ttgl.constexpr = ttgl.NVMMASharedLayout(
            swizzle_byte_width=128, element_bitwidth=16, rank=2, cga_layout=_cga_layout())
        blocked_a: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[1, 0], cga_layout=_cga_layout())
        blocked_b: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        blocked_c: ttgl.constexpr = ttgl.BlockedLayout(
            size_per_thread=[1, 1], threads_per_warp=[32, 1], warps_per_cta=[4, 1],
            order=[0, 1], cga_layout=_cga_layout())
        mma_layout: ttgl.constexpr = ttgl.NVMMADistributedLayout(
            version=[3, 0], warps_per_cta=[4, 1], instr_shape=[16, 32, 16],
            cga_layout=_cga_layout())
        a = ttgl.allocate_shared_memory(ttgl.float16, [block, k], a_layout)
        b = ttgl.allocate_shared_memory(ttgl.float16, [k, block], b_layout)
        a.store(ttgl.full([block, k], 1.0, ttgl.float16, blocked_a))
        if V != 4:
            # variant 4 publishes only operand A generically: the single-
            # publish prelude distinguishes its skeleton from the WGMMA cell
            b.store(ttgl.full([k, block], 1.0, ttgl.float16, blocked_b))
        ttgl.barrier()
        rows = ttgl.arange(0, block, ttgl.SliceLayout(1, blocked_c))[:, None]
        cols = ttgl.arange(0, block, ttgl.SliceLayout(0, blocked_c))[None, :]
        acc = hopper.warpgroup_mma_init(
            ttgl.zeros([block, block], ttgl.float32, mma_layout))
        if V == 0:
            # generic->async operand publish; group consumed under wait(1)
            acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
            acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
        elif V == 1:
            # double consume: the early consume happens under lax wait(1);
            # a correct wait(0)+consume follows, too late for the first read
            acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
            acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
        elif V == 2:
            # explicit proxy fence present, but the only wait is lax wait(2)
            # and the group is still outstanding when the result is consumed
            hopper.fence_async_shared()
            acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
            acc = hopper.warpgroup_mma_wait(num_outstanding=2, deps=[acc])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
        elif V == 3:
            # two generic->async publishes, two groups: wait(1) releases only
            # the oldest but the newest result is consumed first (distinct
            # accumulator seed keeps the two dots from being merged)
            acc2 = hopper.warpgroup_mma_init(
                ttgl.full([block, block], 1.0, ttgl.float32, mma_layout))
            acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
            acc2 = hopper.warpgroup_mma(a, b, acc2, is_async=True)
            acc2 = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc2])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc2, blocked_c))
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
        else:
            # good round then lax round: the second consume happens while its
            # group may still be outstanding, with no trailing completion
            acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
            acc = hopper.warpgroup_mma_wait(num_outstanding=0, deps=[acc])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))
            acc = hopper.warpgroup_mma(a, b, acc, is_async=True)
            acc = hopper.warpgroup_mma_wait(num_outstanding=1, deps=[acc])
            ttgl.store(output + rows * block + cols, ttgl.convert_layout(acc, blocked_c))

    return kernel


def _s2v2_mbarrier_safe_lifecycle():
    @gluon.jit
    def kernel(data, out, BUGGY: ttgl.constexpr):
        # observed-invalidate-reinit lifecycle: a phase is completed and
        # observed, the barrier is invalidated, re-initialized, and driven
        # through a second complete generation -- every transition legal
        bar = mbarrier.allocate_mbarrier()
        mbarrier.init(bar, count=1)
        mbarrier.arrive(bar, count=1)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)
        mbarrier.init(bar, count=1)
        mbarrier.arrive(bar, count=1)
        mbarrier.wait(bar, phase=0)
        mbarrier.invalidate(bar)

    return kernel
