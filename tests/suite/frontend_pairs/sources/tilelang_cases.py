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

    @T.prim_func
    def kernel():
        with T.Kernel(1, threads=threads):
            tx = T.get_thread_binding()
            bar = T.alloc_barrier([1])
            if tx == 0:
                T.mbarrier_arrive(bar)
                if not buggy:
                    T.mbarrier_wait_parity(bar, 0)
                T.mbarrier_arrive(bar)
                if buggy:
                    T.mbarrier_wait_parity(bar, 0)
                T.mbarrier_wait_parity(bar, 1)

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
    width = 16

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
                    Out[0] = S[0]
                T.mbarrier_wait_parity(bar, 0)
            else:
                T.mbarrier_wait_parity(bar, 0)
                if tx == 0:
                    Out[0] = S[0]

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
            A_shared = T.decl_buffer((1024,), T.float16, scope="shared.dyn")
            B_shared = T.decl_buffer((1024,), T.float16, scope="shared.dyn")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.initialize_wgmma_descriptor(
                desc_a,
                T.access_ptr(A_shared[0], "r", 1024),
                2,
                1,
                32,
            )
            T.initialize_wgmma_descriptor(
                desc_b,
                T.access_ptr(B_shared[0], "r", 1024),
                2,
                1,
                32,
            )
            T.warpgroup_fence_operand(C_local, num_regs=32)
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
                T.bool(False),
                1,
                1,
            )
            T.warpgroup_commit_batch()
            if operand_case:
                if buggy and tx == 0:
                    A_shared[0] = T.float16(extent)
                T.warpgroup_wait(0)
                if not buggy and tx == 0:
                    A_shared[0] = T.float16(extent)
                if dual_endpoint and tx == 0:
                    Out[0] = C_local[0]
            else:
                if buggy and tx == 0:
                    Out[0] = C_local[0]
                T.warpgroup_wait(0)
                if not buggy and tx == 0:
                    Out[0] = C_local[0]

    return kernel



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
                else:
                    # TileLang has no public arrive_drop source primitive.
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
            if scenario == "MBARRIER:STALE-GENERATION-WAIT" and buggy:
                T.sync_threads()
                if tx == 1:
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
            if scenario == "TMA-LOAD:GROUPED-BYTES-MISMATCH" or (
                scenario == "TMA-LOAD:MULTI-TRANSFER-ONE-COMPLETION" and buggy
            ):
                T.mbarrier_expect_tx(bar, 16)
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
