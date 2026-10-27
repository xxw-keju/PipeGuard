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


def _suite2_diverse_mbarrier_case(scenario: str, buggy: bool):
    @T.prim_func
    def kernel(A: T.Tensor((256,), T.int32), Out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            count = 2 if scenario in {
                "MBARRIER:MISSING-ARRIVAL",
                "MBARRIER:LATE-MISSING-ARRIVAL",
            } else 1
            bar = T.alloc_barrier([count])
            shared = T.alloc_shared((1,), T.int32)
            if scenario == "MBARRIER:MISSING-ARRIVAL":
                if tx == 0:
                    T.mbarrier_arrive(bar)
                if not buggy and tx == 1:
                    T.mbarrier_arrive(bar)
                if tx < 2:
                    T.mbarrier_wait_parity(bar, 0)
            elif scenario == "MBARRIER:DELAYED-NEXT-ARRIVE-BEFORE-WAIT":
                if tx == 0:
                    shared[0] = 1
                    T.mbarrier_arrive(bar)
                    if not buggy:
                        T.mbarrier_wait_parity(bar, 0)
                    for step in T.serial(64):
                        shared[0] = shared[0] + step + 1
                    T.mbarrier_arrive(bar)
                if tx == 1:
                    T.mbarrier_wait_parity(bar, 1)
                    Out[0] = shared[0]
            elif scenario == "MBARRIER:LATE-MISSING-ARRIVAL":
                if tx < 2:
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                if tx == 0:
                    T.mbarrier_arrive(bar)
                if not buggy and tx == 1:
                    T.mbarrier_arrive(bar)
                if tx < 2:
                    T.mbarrier_wait_parity(bar, 1)
            elif tx == 0:
                if scenario == "MBARRIER:NEXT-ARRIVE-BEFORE-WAIT":
                    T.mbarrier_arrive(bar)
                    if buggy:
                        T.mbarrier_arrive(bar)
                        T.mbarrier_wait_parity(bar, 0)
                        T.mbarrier_wait_parity(bar, 1)
                    else:
                        T.mbarrier_wait_parity(bar, 0)
                        T.mbarrier_arrive(bar)
                        T.mbarrier_wait_parity(bar, 1)
                else:
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 1)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 1 if buggy else 0)

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
    """TMA SYNC_HAZARD cell: five structurally distinct completion-protocol bugs.

    v0 expect_tx overfill: more transaction bytes armed than TMA delivers.
    v1 two transfers, only one transfer's bytes armed on the barrier.
    v2 missing mbarrier arrival: transfer lands but phase never completes.
    v3 expect_tx armed but no arrive on the barrier at all.
    v4 second-phase wait with no second-generation arrival.
    """

    @T.prim_func
    def kernel(A: T.Tensor((64,), T.uint8), Out: T.Tensor((1,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((64,), T.uint8)
            barrier = T.alloc_barrier([32])
            if variant == 0:
                T.mbarrier_expect_tx(barrier, 48)
                T.tma_copy(A[0:32], shared[0:32], barrier=barrier)
                T.mbarrier_arrive(barrier)
                T.mbarrier_wait_parity(barrier, 0)
                if tx == 0:
                    Out[0] = shared[0]
            elif variant == 1:
                T.mbarrier_expect_tx(barrier, 32)
                T.tma_copy(A[0:32], shared[0:32], barrier=barrier)
                T.tma_copy(A[32:64], shared[32:64], barrier=barrier)
                T.mbarrier_arrive(barrier)
                T.mbarrier_wait_parity(barrier, 0)
                if tx == 0:
                    Out[0] = shared[63]
            elif variant == 2:
                T.tma_copy(A[0:32], shared[0:32], barrier=barrier)
                T.mbarrier_wait_parity(barrier, 0)
                if tx == 0:
                    Out[0] = shared[0]
            elif variant == 3:
                T.mbarrier_expect_tx(barrier, 32)
                T.tma_copy(A[0:32], shared[0:32], barrier=barrier)
                T.mbarrier_wait_parity(barrier, 0)
                if tx == 0:
                    Out[0] = shared[0]
            else:
                T.tma_copy(A[0:32], shared[0:32], barrier=barrier)
                T.mbarrier_arrive(barrier)
                T.mbarrier_wait_parity(barrier, 0)
                if tx == 0:
                    Out[0] = shared[0]
                T.mbarrier_wait_parity(barrier, 1)

    return kernel


def _suite2_tma_diverse_case(variant: int, buggy: bool):
    """TMA DATA_HAZARD / SAFE cells: five distinct protocol skeletons each.

    BUG (DATA_HAZARD):
      v0 destination read before the mbarrier wait (classic observe bug).
      v1 wait removed entirely after a barrier-completed load.
      v2 tma store issued, source overwritten before tma_store_wait(read).
      v3 tma store destination read back before tma_store_wait.
      v4 double-buffer loads: waits only buffer0's barrier, reads buffer1.
    CORRECT (SAFE):
      v0 canonical load: transfer + arrive + wait + read.
      v1 two transfers completed through one barrier, then read.
      v2 store chain: write, cta barrier, store, wait(read), reuse source.
      v3 two stores after one cta barrier, single store-wait.
      v4 load -> compute -> store full pipeline.
    """

    @T.prim_func
    def kernel(A: T.Tensor((64,), T.uint8), Out: T.Tensor((64,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((32,), T.uint8)
            if buggy:
                if variant == 0:
                    bar = T.alloc_barrier([32])
                    T.tma_copy(A[0:32], S, barrier=bar)
                    T.mbarrier_arrive(bar)
                    if tx == 0:
                        Out[0] = S[31]
                    T.mbarrier_wait_parity(bar, 0)
                elif variant == 1:
                    bar = T.alloc_barrier([32])
                    T.tma_copy(A[0:32], S, barrier=bar)
                    T.mbarrier_arrive(bar)
                    if tx == 0:
                        Out[0] = S[0]
                elif variant == 2:
                    S[tx] = A[tx]
                    T.sync_threads()
                    T.tma_copy(S, Out[0:32])
                    if tx == 0:
                        S[0] = T.uint8(0)
                    T.tma_store_wait(read=True)
                elif variant == 3:
                    S[tx] = A[tx]
                    T.sync_threads()
                    T.tma_copy(S, Out[0:32])
                    if tx == 0:
                        Out[32] = Out[0]
                    T.tma_store_wait(read=False)
                else:
                    R = T.alloc_shared((32,), T.uint8)
                    bar = T.alloc_barrier([32])
                    bar2 = T.alloc_barrier([32])
                    T.tma_copy(A[0:32], S, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.tma_copy(A[32:64], R, barrier=bar2)
                    T.mbarrier_arrive(bar2)
                    T.mbarrier_wait_parity(bar, 0)
                    if tx == 0:
                        Out[0] = R[0]
            else:
                if variant == 0:
                    bar = T.alloc_barrier([32])
                    T.tma_copy(A[0:32], S, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    if tx == 0:
                        Out[0] = S[0]
                elif variant == 1:
                    R = T.alloc_shared((32,), T.uint8)
                    bar = T.alloc_barrier([32])
                    T.tma_copy(A[0:32], S, barrier=bar)
                    T.tma_copy(A[32:64], R, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    if tx == 0:
                        Out[0] = R[31]
                elif variant == 2:
                    S[tx] = A[tx]
                    T.sync_threads()
                    T.tma_copy(S, Out[0:32])
                    T.tma_store_wait(read=True)
                    if tx == 0:
                        S[0] = T.uint8(0)
                elif variant == 3:
                    R = T.alloc_shared((32,), T.uint8)
                    S[tx] = A[tx]
                    R[tx] = A[32 + tx]
                    T.sync_threads()
                    T.tma_copy(S, Out[0:32])
                    T.tma_copy(R, Out[32:64])
                    T.tma_store_wait(read=False)
                else:
                    R = T.alloc_shared((32,), T.uint8)
                    bar = T.alloc_barrier([32])
                    T.tma_copy(A[0:32], S, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    R[tx] = S[31 - tx] + T.uint8(1)
                    T.sync_threads()
                    T.tma_copy(R, Out[0:32])
                    T.tma_store_wait(read=False)

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
    """WGMMA cells: five structurally distinct completion skeletons per class.

    DATA_HAZARD:
      v0 operand A refilled before warpgroup_wait(0) (kept legacy idiom).
      v1 two gemms, warpgroup_wait(1) retires one, operand refilled anyway.
      v2 tma prefetch into operand A while the gemm group is still in flight.
      v3 accumulator cleared between gemm and its warpgroup_wait.
      v4 operand B refilled between two gemms with no intervening wait.
    SAFE:
      v0 gemm, warpgroup_wait(0), copy accumulator out (kept legacy idiom).
      v1 operand refilled only after warpgroup_wait(0).
      v2 two gemms batched behind a single warpgroup_wait(0).
      v3 shallow wait then full wait before operand reuse.
      v4 two gemm rounds each closed by its own warpgroup_wait(0).
    SYNC_HAZARD:
      v0 accumulator copied out before the warpgroup_wait (kept legacy idiom).
      v1 no warpgroup_wait at all after the gemm.
      v2 warpgroup_wait placed before the gemm it should cover.
      v3 accumulator copied out between two gemms, wait only at the end.
      v4 two gemms but only warpgroup_wait(1) before the accumulator copy.
    """
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
            if data_hazard:
                if variant == 0:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.fill(A_shared, 1.0)
                    T.warpgroup_wait(0)
                    T.copy(C_local, Out)
                elif variant == 1:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(1)
                    T.fill(A_shared, 1.0)
                    T.warpgroup_wait(0)
                    T.copy(C_local, Out)
                elif variant == 2:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.tma_copy(A, A_shared, barrier=barrier_a)
                    T.mbarrier_arrive(barrier_a)
                    T.warpgroup_wait(0)
                    T.mbarrier_wait_parity(barrier_a, 1)
                    T.copy(C_local, Out)
                elif variant == 3:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.clear(C_local)
                    T.warpgroup_wait(0)
                    T.copy(C_local, Out)
                else:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.fill(B_shared, 2.0)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                    T.copy(C_local, Out)
            elif sync_hazard:
                if variant == 0:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                    T.warpgroup_wait(0)
                elif variant == 1:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                elif variant == 2:
                    T.warpgroup_wait(0)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                elif variant == 3:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                else:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(1)
                    T.copy(C_local, Out)
            else:
                if variant == 0:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                    T.copy(C_local, Out)
                elif variant == 1:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                    T.fill(A_shared, 1.0)
                    T.copy(C_local, Out)
                elif variant == 2:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                    T.copy(C_local, Out)
                elif variant == 3:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(1)
                    T.warpgroup_wait(0)
                    T.fill(A_shared, 1.0)
                    T.copy(C_local, Out)
                else:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                    T.copy(C_local, Out)

    return kernel


def _suite2_proxy_case(variant: int, result_class: str):
    """Proxy cells: five structurally distinct generic<->async proxy skeletons per class.

    DATA_HAZARD (InjectFenceProxy disabled by the runner for this class):
      v0 generic fill then unfenced tma store (kept legacy idiom).
      v1 one fence, then a second unfenced generation of fill + tma store.
      v2 generic fill racing an in-flight tma load of the same buffer.
      v3 generic global write then unfenced tma load reading it back.
      v4 fragment-staged copy into shared, then unfenced tma store.
    SAFE:
      v0 fill, fence_proxy_async, tma store, store-wait (kept legacy idiom).
      v1 fill, fence, tma load overwriting the same buffer, wait, read.
      v2 one fence publishing two buffers to two tma stores.
      v3 fence + store + store-wait(read) then generic reuse of the source.
      v4 load -> generic compute -> fence -> store round trip.
    SYNC_HAZARD (wgmma pipeline behind a proxy fence):
      v0 accumulator copied out before the final warpgroup_wait (kept legacy idiom).
      v1 no warpgroup_wait after the fenced gemm.
      v2 warpgroup_wait before the gemm it should cover.
      v3 copy between two gemms; wait only at the end.
      v4 two gemms with only warpgroup_wait(1) before the copy.
    """
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
                if variant == 0:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                    T.warpgroup_wait(0)
                elif variant == 1:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                elif variant == 2:
                    T.warpgroup_wait(0)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                elif variant == 3:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.copy(C_local, Out)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(0)
                else:
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.wgmma_gemm(A_shared, B_shared, C_local)
                    T.warpgroup_wait(1)
                    T.copy(C_local, Out)
        return sync_kernel

    data_hazard = result_class == "DATA_HAZARD"

    if result_class == "SAFE" and variant == 2:
        @T.prim_func
        def flat_store_kernel(OutFlat: T.Tensor((4096,), T.float16)):
            with T.Kernel(1, threads=128):
                s1 = T.alloc_shared((2048,), T.float16)
                s2 = T.alloc_shared((2048,), T.float16)
                T.fill(s1, T.float16(1))
                T.fill(s2, T.float16(2))
                T.fence_proxy_async()
                T.tma_copy(s1, OutFlat[0:2048])
                T.tma_copy(s2, OutFlat[2048:4096])
                T.tma_store_wait(read=False)
        return flat_store_kernel

    @T.prim_func
    def store_kernel(Out: T.Tensor((64, 64), T.float16)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((64, 64), T.float16)
            if data_hazard:
                if variant == 0:
                    T.fill(shared, T.float16(1))
                    T.tma_copy(shared, Out)
                    T.tma_store_wait(read=False)
                elif variant == 1:
                    T.fill(shared, T.float16(1))
                    T.fence_proxy_async()
                    T.tma_copy(shared, Out)
                    T.tma_store_wait(read=False)
                    T.fill(shared, T.float16(2))
                    T.tma_copy(shared, Out)
                    T.tma_store_wait(read=False)
                elif variant == 2:
                    staging = T.alloc_shared((64, 64), T.float16)
                    bar = T.alloc_barrier([128])
                    T.tma_copy(Out, shared, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.fill(shared, T.float16(2))
                    T.mbarrier_wait_parity(bar, 0)
                    T.copy(shared, staging)
                elif variant == 3:
                    staging = T.alloc_shared((64, 64), T.float16)
                    bar = T.alloc_barrier([128])
                    tx = T.get_thread_binding()
                    if tx == 0:
                        Out[0, 0] = T.float16(1)
                    T.tma_copy(Out, shared, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.copy(shared, staging)
                else:
                    frag = T.alloc_fragment((64, 64), T.float16)
                    T.clear(frag)
                    T.copy(frag, shared)
                    T.tma_copy(shared, Out)
                    T.tma_store_wait(read=False)
            else:
                if variant == 0:
                    T.fill(shared, T.float16(1))
                    T.fence_proxy_async()
                    T.tma_copy(shared, Out)
                    T.tma_store_wait(read=False)
                elif variant == 1:
                    staging = T.alloc_shared((64, 64), T.float16)
                    bar = T.alloc_barrier([128])
                    T.fill(shared, T.float16(1))
                    T.fence_proxy_async()
                    T.tma_copy(Out, shared, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.copy(shared, staging)
                elif variant == 3:
                    staging = T.alloc_shared((64, 64), T.float16)
                    T.fill(shared, T.float16(1))
                    T.fence_proxy_async()
                    T.tma_copy(shared, Out)
                    T.tma_store_wait(read=True)
                    T.copy(shared, staging)
                else:
                    staging = T.alloc_shared((64, 64), T.float16)
                    bar = T.alloc_barrier([128])
                    T.tma_copy(Out, shared, barrier=bar)
                    T.mbarrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    T.copy(shared, staging)
                    T.fence_proxy_async()
                    T.tma_copy(staging, Out)
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
    if family == "MBARRIER-DIVERSE":
        return _suite2_diverse_mbarrier_case(scenario_id, buggy)
    if family == "MB-REUSE":
        return _mbarrier_case(variant, buggy)
    if family in {"CP-WAIT", "CP-OVERLAP"}:
        return _cp_async_case(family, variant, buggy)
    if family == "TMA-OBSERVE":
        return _suite2_tma_diverse_case(variant, buggy)
    if family == "TMA-PROXY":
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
    """cp.async cells: five structurally distinct group-protocol skeletons per class.

    DATA_HAZARD:
      v0 generic store into the in-flight destination before wait_group(0).
      v1 commit but no wait_group; cta barrier does not complete the group.
      v2 wait_group(1) leaves the single committed group in flight; read races.
      v3 two groups, wait_group(1) retires the older; newest destination overwritten.
      v4 two overlapping destinations inside one commit group (async WAW).
    SAFE:
      v0 issue, commit, wait_group(0), then generic store + read (canonical).
      v1 two groups, wait_group(1), touch only the retired group's destination.
      v2 per-thread copies, wait_group(0) + cta barrier, cross-thread read.
      v3 two disjoint destinations in one group, wait, read.
      v4 two-stage prefetch pipeline consuming stage k under wait_group(1)/(0).
    SYNC_HAZARD:
      v0 wait_group issued before the commit it should cover.
      v1 second copy issued but never committed; wait_group(0) cannot cover it.
      v2 completion handoff via divergent mbarrier (half the arrivals missing).
      v3 commit issued only after the wait that should have observed it.
      v4 commit under a divergent guard: half the copies stay uncommitted.
    """

    @T.prim_func
    def kernel(A: T.Tensor((128,), T.uint8), Out: T.Tensor((32,), T.uint8)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            S = T.alloc_shared((128,), T.uint8)
            if result_class == "DATA_HAZARD":
                if variant == 0:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                        S[0] = T.uint8(1)
                        T.ptx_wait_group(0)
                        Out[0] = S[0]
                elif variant == 1:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                    T.sync_threads()
                    if tx == 1:
                        Out[0] = S[0]
                elif variant == 2:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_wait_group(1)
                        Out[0] = S[0]
                elif variant == 3:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_cp_async(T.access_ptr(S[16], "w", 16), T.access_ptr(A[16], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_wait_group(1)
                        S[16] = T.uint8(7)
                        T.ptx_wait_group(0)
                        Out[0] = S[16]
                else:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_cp_async(T.access_ptr(S[8], "w", 16), T.access_ptr(A[16], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        Out[0] = S[8]
            elif result_class == "SAFE":
                if variant == 0:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        S[0] = T.uint8(1)
                        Out[0] = S[0]
                elif variant == 1:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_cp_async(T.access_ptr(S[16], "w", 16), T.access_ptr(A[16], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_wait_group(1)
                        S[0] = T.uint8(7)
                        T.ptx_wait_group(0)
                        Out[0] = S[16]
                elif variant == 2:
                    T.ptx_cp_async(T.access_ptr(S[tx * 4], "w", 4), T.access_ptr(A[tx * 4], "r", 4), 4)
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    Out[tx] = S[127 - tx * 4]
                elif variant == 3:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_cp_async(T.access_ptr(S[16], "w", 16), T.access_ptr(A[16], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_wait_group(0)
                        Out[0] = S[16]
                else:
                    acc = T.alloc_shared((1,), T.uint8)
                    if tx == 0:
                        acc[0] = T.uint8(0)
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_cp_async(T.access_ptr(S[16], "w", 16), T.access_ptr(A[16], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_wait_group(1)
                        acc[0] = S[0]
                        T.ptx_wait_group(0)
                        acc[0] = acc[0] + S[16]
                        Out[0] = acc[0]
            else:
                if variant == 0:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_wait_group(0)
                        S[0] = T.uint8(1)
                        Out[0] = S[0]
                elif variant == 1:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_commit_group()
                        T.ptx_cp_async(T.access_ptr(S[16], "w", 16), T.access_ptr(A[16], "r", 16), 16)
                        T.ptx_wait_group(0)
                        Out[0] = S[16]
                elif variant == 2:
                    bar = T.alloc_barrier(32)
                    T.ptx_cp_async(T.access_ptr(S[tx * 4], "w", 4), T.access_ptr(A[tx * 4], "r", 4), 4)
                    T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    if tx < 16:
                        T.barrier_arrive(bar)
                    T.mbarrier_wait_parity(bar, 0)
                    Out[tx] = S[127 - tx * 4]
                elif variant == 3:
                    if tx == 0:
                        T.ptx_cp_async(T.access_ptr(S[0], "w", 16), T.access_ptr(A[0], "r", 16), 16)
                        T.ptx_wait_group(0)
                        T.ptx_commit_group()
                        Out[0] = S[0]
                else:
                    T.ptx_cp_async(T.access_ptr(S[tx * 4], "w", 4), T.access_ptr(A[tx * 4], "r", 4), 4)
                    if tx < 16:
                        T.ptx_commit_group()
                    T.ptx_wait_group(0)
                    T.sync_threads()
                    Out[tx] = S[127 - tx * 4]

    return kernel
