# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-memory-triton-safe-02
# independent_label=SAFE; idiom=adjacent-disjoint WAR: read and write touch adjacent but non-overlapping elements
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    value = ttgl.load(Data + lanes * 0 + 24, mask=lanes == 0, other=0)
    ttgl.store(Out + lanes * 0 + 1, value, mask=lanes == 0)
    ttgl.store(Data + lanes * 0 + 25, 9, mask=lanes == 64)
