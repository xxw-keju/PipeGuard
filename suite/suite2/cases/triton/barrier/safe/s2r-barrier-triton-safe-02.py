# Generated Suite 2 Barrier control; do not hand edit.
# case_id=s2r-barrier-triton-safe-02
# scenario=BARRIER:SHARED-WAR; independent_label=SAFE
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
    shared.store(ttgl.full([128], 7, ttgl.int32, _layout()))
    ttgl.barrier()
    sink.store(shared.load(_layout()))
    ttgl.barrier()
    shared.store(ttgl.full([128], 202, ttgl.int32, _layout()))
