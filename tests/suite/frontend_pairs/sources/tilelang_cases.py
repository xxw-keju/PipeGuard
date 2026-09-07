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


def _wgmma_case(variant: int, buggy: bool):
    extent = 64
    operand_case = variant % 2 == 0

    @T.prim_func
    def kernel(Out: T.Tensor((1,), T.float16)):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            C_local[0] = T.float16(0)
            T.initialize_wgmma_descriptor(desc_a, T.uint64(0), 2, 1, 32)
            T.initialize_wgmma_descriptor(desc_b, T.uint64(0), 2, 1, 32)
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
            else:
                if buggy and tx == 0:
                    Out[0] = C_local[0]
                T.warpgroup_wait(0)
                if not buggy and tx == 0:
                    Out[0] = C_local[0]

    return kernel


def build_tilelang_case(family: str, variant: int, polarity: str):
    buggy = polarity == "BUG"
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


def lower_tilelang_case(family: str, variant: int, polarity: str):
    function = build_tilelang_case(family, variant, polarity)
    target = tvm.target.Target({
        "kind": "cuda",
        "arch": "sm_90a",
        "thread_warp_size": 32,
        "max_num_threads": 1024,
    })
    pass_config = (
        {"tl.disable_thread_storage_sync": True}
        if family == "CTA-HANDOFF" else {}
    )
    with target, tvm.transform.PassContext(opt_level=3, config=pass_config):
        return tilelang.lower(
            function,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        ).device_mod


def write_tilelang_case(path: str | Path, family: str, variant: int, polarity: str) -> None:
    Path(path).write_text(tvm.ir.save_json(lower_tilelang_case(family, variant, polarity)))
