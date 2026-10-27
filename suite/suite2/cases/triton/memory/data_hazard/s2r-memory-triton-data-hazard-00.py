# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-memory-triton-data-hazard-00
# independent_label=DATA_HAZARD; idiom=cross-warp WAW: two unordered writes of distinct values to one global address
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    ttgl.store(Data + lanes * 0 + 8, 101, mask=lanes == 0)
    ttgl.store(Data + lanes * 0 + 8, 202, mask=lanes == 64)
