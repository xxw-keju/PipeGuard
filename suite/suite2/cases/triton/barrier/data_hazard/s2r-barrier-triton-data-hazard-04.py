# Generated Suite 2 Barrier case; do not hand edit.
# case_id=s2r-barrier-triton-data-hazard-04
# scenario=BARRIER:GLOBAL-RAW; independent_label=DATA_HAZARD
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
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    ttgl.store(Data + lanes * 0 + 8, 101, mask=lanes == 0)
    value = ttgl.load(Data + lanes * 0 + 8, mask=lanes == 64, other=0)
    ttgl.store(Out + lanes * 0 + 5, value, mask=lanes == 64)
