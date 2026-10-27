# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-memory-triton-data-hazard-03
# independent_label=DATA_HAZARD; idiom=same-value WAW plus third-party reader: CUDA memory model treats unordered same-address writes as a race even when values agree
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    ttgl.store(Data + lanes * 0 + 8, 55, mask=lanes == 0)
    ttgl.store(Data + lanes * 0 + 8, 55, mask=lanes == 64)
    value = ttgl.load(Data + lanes * 0 + 8, mask=lanes == 32, other=0)
    ttgl.store(Out + lanes * 0 + 2, value, mask=lanes == 32)
