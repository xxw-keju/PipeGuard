# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-memory-triton-safe-04
# independent_label=SAFE; idiom=same-thread double-buffer alternation: one thread ping-pongs between two slots in program order
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    ttgl.store(Data + lanes * 0 + 40, 1, mask=lanes == 64)
    first = ttgl.load(Data + lanes * 0 + 40, mask=lanes == 64, other=0)
    ttgl.store(Data + lanes * 0 + 44, first + 1, mask=lanes == 64)
    second = ttgl.load(Data + lanes * 0 + 44, mask=lanes == 64, other=0)
    ttgl.store(Out + lanes * 0 + 3, second, mask=lanes == 64)
