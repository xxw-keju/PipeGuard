# Generated Suite 2 Barrier control; do not hand edit.
# case_id=s2r-barrier-triton-safe-03
# scenario=BARRIER:SHARED-DELAYED-RAW; independent_label=SAFE
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.constexpr_function
def _shared_layout():
    return ttgl.SwizzledSharedLayout(vec=1, per_phase=1, max_phase=1, order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    shared = ttgl.allocate_shared_memory(ttgl.int32, [128], _shared_layout())
    sink = ttgl.allocate_shared_memory(ttgl.int32, [128], _shared_layout())
    shared.store(ttgl.full([128], 101, ttgl.int32, _layout()))
    for step in ttgl.static_range(0, 64):
        sink.store(ttgl.full([128], step, ttgl.int32, _layout()))
    ttgl.barrier()
    sink.store(shared.load(_layout()))
