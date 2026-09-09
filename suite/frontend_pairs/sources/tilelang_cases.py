from pathlib import Path

import tilelang
import tilelang.language as T
from tilelang import tvm


def _shape(variant: int) -> tuple[int, int, int]:
    elements = (1, 2, 4, 8)[variant % 4]
    stages = (2, 3, 4)[variant % 3]
    threads = (32, 64)[variant % 2]
    return elements, stages, threads


def _memory_case(family: str, variant: int, buggy: bool):
    elements, _stages, threads = _shape(variant)
    size = max(16, elements + 1)

    @T.prim_func
    def kernel(A: T.Tensor((size,), T.int32), Out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            shared = T.decl_buffer((1,), T.int32, scope="shared.dyn")
            if family == "CTA-HANDOFF":
                if tx == 0:
                    shared[0] = 1
                if not buggy:
                    T.sync_threads()
                if tx == 1:
                    Out[0] = shared[0]
            else:
                if tx == 0:
                    A[0] = 1
                if tx == 1:
                    index = 0 if buggy else elements
                    if family == "MEM-WAW":
                        A[index] = 2
                    else:
                        Out[0] = A[index]

    return kernel


def _mbarrier_case(variant: int, buggy: bool):
    _elements, _stages, threads = _shape(variant)
    protocol_depth = variant + 2

    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            bar = T.alloc_barrier([1])
            if tx == 0:
                for phase in T.serial(protocol_depth):
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, phase % 2)
                if buggy:
                    T.mbarrier_arrive(bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, protocol_depth % 2)
                    T.mbarrier_wait_parity(bar, (protocol_depth + 1) % 2)

    return kernel


def _cp_async_case(family: str, variant: int, buggy: bool):
    elements, _stages, threads = _shape(variant)
    byte_count = (4, 8, 16)[variant % 3]
    size = max(32, 2 * byte_count)

    @T.prim_func
    def kernel(A: T.Tensor((size,), T.uint8), Out: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            S = T.alloc_shared((size,), T.uint8)
            if tx == 0:
                T.ptx_cp_async(
                    T.access_ptr(S[0], "w", byte_count),
                    T.access_ptr(A[0], "r", byte_count),
                    byte_count,
                )
                if family == "CP-OVERLAP":
                    destination = 0 if buggy else byte_count
                    T.ptx_cp_async(
                        T.access_ptr(S[destination], "w", byte_count),
                        T.access_ptr(A[byte_count], "r", byte_count),
                        byte_count,
                    )
                T.ptx_commit_group()
                if family == "CP-WAIT" and buggy:
                    S[0] = T.uint8(elements)
                T.ptx_wait_group(0)
                if family == "CP-WAIT" and not buggy:
                    S[0] = T.uint8(elements)
                Out[0] = S[0]

    return kernel


def _tma_load_case(family: str, variant: int, buggy: bool):
    elements, _stages, threads = _shape(variant)
    width = 16 * (variant + 1)

    @T.prim_func
    def kernel(A: T.Tensor((width,), T.uint8), Out: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            S = T.alloc_shared((width,), T.uint8)
            bar = T.alloc_barrier([threads])
            if family == "TMA-PROXY" and tx == 0:
                A[0] = T.uint8(elements)
            if family == "TMA-PROXY" and not buggy:
                T.fence_proxy_async()
            T.tma_copy(A[0:width], S, barrier=bar)
            T.mbarrier_arrive(bar)
            if family == "TMA-OBSERVE" and buggy:
                if tx == 0:
                    Out[0] = S[width - 1]
                T.mbarrier_wait_parity(bar, 0)
            else:
                T.mbarrier_wait_parity(bar, 0)
                if tx == 0:
                    Out[0] = S[0]

    return kernel


def _suite2_tma_sync_case(variant: int):
    width = 16 * (variant + 1)

    @T.prim_func
    def kernel(A: T.Tensor((width,), T.uint8), Out: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((width,), T.uint8)
            barrier = T.alloc_barrier([32])
            T.mbarrier_expect_tx(barrier, width + 16)
            T.tma_copy(A[0:width], shared, barrier=barrier)
            T.mbarrier_arrive(barrier)
            T.mbarrier_wait_parity(barrier, 0)
            if tx == 0:
                Out[0] = shared[0]

    return kernel


def _tma_store_case(variant: int, buggy: bool):
    elements, _stages, threads = _shape(variant)
    width = 16
    source_case = variant % 2 == 0

    @T.prim_func
    def kernel(Out: T.Tensor((width,), T.uint8), Sink: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            S = T.alloc_shared((width,), T.uint8)
            if tx == 0:
                S[0] = T.uint8(elements)
            T.sync_threads()
            T.tma_copy(S, Out[0:width])
            if source_case:
                if buggy and tx == 0:
                    S[0] = T.uint8(0)
                T.tma_store_wait(read=True)
                if not buggy and tx == 0:
                    S[0] = T.uint8(0)
            else:
                T.tma_store_wait(read=True)
                if buggy and tx == 0:
                    Sink[0] = Out[0]
                T.tma_store_wait(read=False)
                if not buggy and tx == 0:
                    Sink[0] = Out[0]

    return kernel


def _wgmma_case(
    variant: int,
    buggy: bool,
    *,
    threads: int = 128,
    omit_arrive: bool = False,
    dual_endpoint: bool = False,
):
    extent = 64
    operand_case = variant % 2 == 0 or dual_endpoint

    @T.prim_func
    def kernel(Out: T.Tensor((1,), T.float16)):
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            C_local[0] = T.float16(0)
            T.initialize_wgmma_descriptor(desc_a, T.uint64(0), 2, 1, 32)
            T.initialize_wgmma_descriptor(desc_b, T.uint64(0), 2, 1, 32)
            if not omit_arrive:
                T.warpgroup_arrive()
            T.fence_proxy_async()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )
            T.warpgroup_commit_batch()
            if operand_case:
                if buggy and tx == 0:
                    A_shared[0] = T.uint8(extent)
                T.warpgroup_wait(0)
                if not buggy and tx == 0:
                    A_shared[0] = T.uint8(extent)
                if dual_endpoint and tx == 0:
                    Out[0] = C_local[0]
            else:
                if buggy and tx == 0:
                    Out[0] = C_local[0]
                T.warpgroup_wait(0)
                if not buggy and tx == 0:
                    Out[0] = C_local[0]

    return kernel


def _suite2_wgmma_case(variant: int, result_class: str):
    repeats = variant + 1
    data_hazard = result_class == "DATA_HAZARD"
    sync_hazard = result_class == "SYNC_HAZARD"

    @T.prim_func
    def kernel(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        Out: T.Tensor((64, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            barrier_a = T.alloc_barrier([128])
            barrier_b = T.alloc_barrier([128])
            T.tma_copy(A, A_shared, barrier=barrier_a)
            T.mbarrier_arrive(barrier_a)
            T.tma_copy(B, B_shared, barrier=barrier_b)
            T.mbarrier_arrive(barrier_b)
            T.mbarrier_wait_parity(barrier_a, 0)
            T.mbarrier_wait_parity(barrier_b, 0)
            T.clear(C_local)
            for _phase in T.serial(repeats - 1):
                T.wgmma_gemm(A_shared, B_shared, C_local)
                T.warpgroup_wait(0)
            T.wgmma_gemm(A_shared, B_shared, C_local)
            if data_hazard:
                T.fill(A_shared, 1.0)
                T.warpgroup_wait(0)
                T.copy(C_local, Out)
            elif sync_hazard:
                T.copy(C_local, Out)
                T.warpgroup_wait(0)
            else:
                T.warpgroup_wait(0)
                T.copy(C_local, Out)

    return kernel


def _suite2_proxy_case(variant: int, result_class: str):
    repeats = variant + 1
    if result_class == "SYNC_HAZARD":
        @T.prim_func
        def sync_kernel(
            A: T.Tensor((64, 16), T.float16),
            B: T.Tensor((16, 64), T.float16),
            Out: T.Tensor((64, 64), T.float32),
        ):
            with T.Kernel(1, threads=128):
                A_shared = T.alloc_shared((64, 16), T.float16)
                B_shared = T.alloc_shared((16, 64), T.float16)
                C_local = T.alloc_fragment((64, 64), T.float32)
                barrier_a = T.alloc_barrier([128])
                barrier_b = T.alloc_barrier([128])
                T.tma_copy(A, A_shared, barrier=barrier_a)
                T.mbarrier_arrive(barrier_a)
                T.tma_copy(B, B_shared, barrier=barrier_b)
                T.mbarrier_arrive(barrier_b)
                T.mbarrier_wait_parity(barrier_a, 0)
                T.mbarrier_wait_parity(barrier_b, 0)
                T.fill(A_shared, 1.0)
                T.fence_proxy_async()
                T.clear(C_local)
                for _phase in T.serial(repeats - 1):
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                T.wgmma_gemm(A_shared, B_shared, C_local)
                T.copy(C_local, Out)
                T.warpgroup_wait(0)
        return sync_kernel

    data_hazard = result_class == "DATA_HAZARD"

    @T.prim_func
    def store_kernel(Out: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            for phase in T.serial(repeats):
                T.fill(shared, T.float16(phase + 1))
            if not data_hazard:
                T.fence_proxy_async()
            T.tma_copy(shared, Out)
            T.tma_store_wait(read=False)

    return store_kernel


def _named_memory_case(scenario: str, buggy: bool):
    @T.prim_func
    def kernel(OutA: T.Tensor((64,), T.uint8), OutB: T.Tensor((64,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((64,), T.uint8)
            if scenario == "MEMORY:PARTIAL-WAR":
                for i in T.serial(4):
                    if tx == 0:
                        OutA[i] = shared[i]
                    if tx == 1:
                        shared[i + 2] = T.uint8(1)
            elif scenario == "MEMORY:ADJACENT-DISJOINT":
                if tx == 0:
                    shared[0] = T.uint8(1)
                if tx == 1:
                    OutA[1] = shared[16]
            elif scenario == "MEMORY:DANGLING-REGION":
                # Native IR cannot contain a dangling allocation reference.
                # This real out-of-bounds operation must be rejected or reported
                # as not equivalent to the model-only direct PGIR case.
                if tx == 0:
                    shared[64] = T.uint8(1)
                if tx == 1:
                    OutA[1] = shared[64]
            elif scenario == "MEMORY:CONTAINED-WAW":
                for i in T.serial(16):
                    if tx == 0:
                        shared[i] = T.uint8(1)
                for i in T.serial(4):
                    if tx == 1:
                        shared[i + (4 if buggy else 16)] = T.uint8(2)
            elif scenario == "MEMORY:ALLOCATION-IDENTITY":
                if tx == 0:
                    OutA[0] = T.uint8(1)
                if tx == 1:
                    if buggy:
                        OutA[0] = T.uint8(2)
                    else:
                        OutB[0] = T.uint8(2)

    return kernel


def _named_cta_case(scenario: str, buggy: bool):
    @T.prim_func
    def kernel(Out: T.Tensor((2,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((8,), T.uint8)
            if scenario == "CTA-BARRIER:MISSING-PARTICIPANT":
                if buggy:
                    if tx == 0:
                        T.sync_threads()
                else:
                    T.sync_threads()
            elif scenario == "CTA-BARRIER:DUPLICATE-ARRIVAL":
                T.sync_threads()
                if buggy and tx == 0:
                    T.sync_threads()
            else:
                if tx == 0:
                    shared[0] = T.uint8(1)
                if scenario == "CTA-BARRIER:COMPLETE-HANDOFF" or not buggy:
                    T.sync_threads()
                if tx == 1:
                    Out[1] = shared[0]

    return kernel


def _named_mbarrier_case(scenario: str, buggy: bool):
    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if scenario == "MBARRIER:USE-BEFORE-INIT":
                bar = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            else:
                bar = T.alloc_barrier([1])
            if tx == 0:
                if scenario == "MBARRIER:TX-BYTES-MISMATCH":
                    T.mbarrier_expect_tx(bar, 8 if buggy else 16)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                elif scenario == "MBARRIER:NEXT-ARRIVE-BEFORE-WAIT":
                    T.mbarrier_arrive(bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.mbarrier_wait_parity(bar, 1)
                elif scenario == "MBARRIER:INVALIDATE-REQUIRES-OBSERVATION":
                    T.mbarrier_arrive(bar)
                    if not buggy:
                        T.mbarrier_wait_parity(bar, 0)
                elif scenario == "MBARRIER:REINITIALIZE-LIFECYCLE":
                    T.mbarrier_arrive(bar)
                    if not buggy:
                        T.mbarrier_wait_parity(bar, 0)
                elif scenario in {
                    "MBARRIER:NONCOALESCED-POLL",
                    "MBARRIER:TWO-GENERATION-HANDOFF",
                }:
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 1)
                elif scenario == "MBARRIER:PARITY-WRAP":
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 1)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 1 if buggy else 0)
                elif scenario == "MBARRIER:STALE-GENERATION-WAIT":
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 1)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    if buggy:
                        T.mbarrier_wait_parity(bar, 0)
                else:
                    # TileLang has no public arrive_drop source primitive.
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)

    return kernel


def _named_cp_async_case(scenario: str, buggy: bool):
    @T.prim_func
    def kernel(A: T.Tensor((64,), T.uint8), Out: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((64,), T.uint8)
            if tx == 0:
                destination = A if scenario == "CP-ASYNC:INVALID-DESTINATION-SPACE" else S
                T.ptx_cp_async(
                    T.access_ptr(destination[0], "w", 16),
                    T.access_ptr(A[0], "r", 16),
                    16,
                )
                if scenario == "CP-ASYNC:WAIT-BEFORE-COMMIT":
                    T.ptx_wait_group(0)
                    S[0] = T.uint8(1)
                elif scenario == "CP-ASYNC:TWO-GROUP-WAIT-ONE":
                    T.ptx_commit_group()
                    T.ptx_cp_async(
                        T.access_ptr(S[16], "w", 16),
                        T.access_ptr(A[16], "r", 16),
                        16,
                    )
                    T.ptx_commit_group()
                    T.ptx_wait_group(1)
                    S[16 if buggy else 0] = T.uint8(2)
                    T.ptx_wait_group(0)
                else:
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    S[0] = T.uint8(1)
                Out[0] = S[0]

    return kernel


def _named_tma_load_case(scenario: str, buggy: bool):
    @T.prim_func
    def kernel(A: T.Tensor((32,), T.uint8), Out: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((32,), T.uint8)
            bar = T.alloc_barrier([32])
            if scenario in {"TMA-LOAD:GENERIC-SOURCE-NO-PROXY", "TMA-LOAD:PROXY-AND-WAIT"} and tx == 0:
                A[0] = T.uint8(1)
            if scenario == "TMA-LOAD:PROXY-AND-WAIT":
                T.fence_proxy_async()
            if scenario in {"TMA-LOAD:GROUPED-BYTES-MISMATCH", "TMA-LOAD:MULTI-TRANSFER-ONE-COMPLETION"}:
                T.mbarrier_expect_tx(bar, 16 if buggy else 32)
            T.tma_copy(A[0:16], S[0:16], barrier=bar)
            if scenario == "TMA-LOAD:MULTI-TRANSFER-ONE-COMPLETION":
                T.tma_copy(A[16:32], S[16:32], barrier=bar)
            T.mbarrier_arrive(bar)
            if scenario == "TMA-LOAD:DESTINATION-BEFORE-WAIT" and buggy:
                if tx == 0:
                    Out[0] = S[0]
                T.mbarrier_wait_parity(bar, 0)
            else:
                T.mbarrier_wait_parity(bar, 0)
                if tx == 0:
                    Out[0] = S[0]

    return kernel


def _named_tma_store_case(scenario: str, buggy: bool):
    @T.prim_func
    def kernel(Out: T.Tensor((32,), T.uint8), Sink: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((32,), T.uint8)
            if tx == 0:
                S[0] = T.uint8(1)
            T.sync_threads()
            if scenario == "TMA-STORE:STORE-WAIT-BEFORE-COMMIT":
                T.tma_store_wait(read=True)
                T.tma_copy(S[0:16], Out[0:16])
                if tx == 0:
                    S[0] = T.uint8(0)
            elif scenario == "TMA-STORE:STORE-INVALID-SOURCE":
                T.tma_copy(Out[0:16], Out[16:32])
            elif scenario == "TMA-STORE:TWO-GROUP-WAIT-ONE":
                T.tma_copy(S[16:32], Out[16:32])
                T.tma_store_wait(count=1, read=False)
                if tx == 0:
                    Sink[0] = Out[16 if buggy else 0]
                T.tma_store_wait(count=0, read=False)
            else:
                T.tma_store_wait(read=True)
                if tx == 0:
                    S[0] = T.uint8(0)
                T.tma_store_wait(read=False)
                if tx == 0:
                    Sink[0] = Out[0]

    return kernel

def build_tilelang_case(family: str, scenario_id: str, variant: int, polarity: str):
    buggy = polarity == "BUG"
    if family == "MEMORY":
        return _named_memory_case(scenario_id, buggy)
    if family == "CTA-BARRIER":
        return _named_cta_case(scenario_id, buggy)
    if family == "MBARRIER":
        return _named_mbarrier_case(scenario_id, buggy)
    if family == "CP-ASYNC":
        return _named_cp_async_case(scenario_id, buggy)
    if family == "TMA-LOAD":
        return _named_tma_load_case(scenario_id, buggy)
    if family == "WGMMA":
        return _wgmma_case(
            variant,
            buggy,
            threads=64 if scenario_id == "WGMMA:INVALID-WARPGROUP-SHAPE" else 128,
            omit_arrive=scenario_id == "WGMMA:MISSING-OPERAND-FENCE" and buggy,
            dual_endpoint=scenario_id == "WGMMA:DUAL-ENDPOINT",
        )
    if family == "TMA-STORE":
        return _named_tma_store_case(scenario_id, buggy)
    if family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        return _memory_case(family, variant, buggy)
    if family == "MB-REUSE":
        return _mbarrier_case(variant, buggy)
    if family in {"CP-WAIT", "CP-OVERLAP"}:
        return _cp_async_case(family, variant, buggy)
    if family in {"TMA-OBSERVE", "TMA-PROXY"}:
        return _tma_load_case(family, variant, buggy)
    if family == "WGMMA-ENDPOINT":
        return _wgmma_case(variant, buggy)
    if family == "TMA-STORE-ENDPOINT":
        return _tma_store_case(variant, buggy)
    raise KeyError(f"unsupported TileLang frontend-pair family: {family}")


def lower_tilelang_case(family: str, scenario_id: str, variant: int, polarity: str):
    function = build_tilelang_case(family, scenario_id, variant, polarity)
    target = tvm.target.Target({
        "kind": "cuda",
        "arch": "sm_90a",
        "thread_warp_size": 32,
        "max_num_threads": 1024,
    })
    pass_config = (
        {"tl.disable_thread_storage_sync": True}
        if family in {"CTA-HANDOFF", "CTA-BARRIER"} else {}
    )
    with target, tvm.transform.PassContext(opt_level=3, config=pass_config):
        return tilelang.lower(
            function,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        ).device_mod


def write_tilelang_case(
    path: str | Path,
    family: str,
    scenario_id: str,
    variant: int,
    polarity: str,
) -> None:
    Path(path).write_text(tvm.ir.save_json(
        lower_tilelang_case(family, scenario_id, variant, polarity)
    ))


def _suite2_cp_async_case(variant: int, result_class: str):
    groups = 9 + variant
    bytes_per_group = 16
    size = groups * bytes_per_group

    @T.prim_func
    def kernel(A: T.Tensor((size,), T.uint8), Out: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((size,), T.uint8)
            if tx == 0:
                for group in T.unroll(8):
                    offset = group * bytes_per_group
                    T.ptx_cp_async(
                        T.access_ptr(S[offset], "w", bytes_per_group),
                        T.access_ptr(A[offset], "r", bytes_per_group),
                        bytes_per_group,
                    )
                    T.ptx_commit_group()
                if result_class != "SYNC_HAZARD":
                    T.ptx_wait_group(0)
                for tail_group in T.unroll(groups - 8):
                    offset = (tail_group + 8) * bytes_per_group
                    T.ptx_cp_async(
                        T.access_ptr(S[offset], "w", bytes_per_group),
                        T.access_ptr(A[offset], "r", bytes_per_group),
                        bytes_per_group,
                    )
                    T.ptx_commit_group()
                final_offset = (groups - 1) * bytes_per_group
                if result_class == "DATA_HAZARD":
                    S[final_offset] = T.uint8(variant + 1)
                T.ptx_wait_group(0)
                if result_class == "SAFE":
                    S[final_offset] = T.uint8(variant + 1)
                Out[0] = S[final_offset]

    return kernel
