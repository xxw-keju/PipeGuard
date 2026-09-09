# Generated Recovery Suite 2 real Triton Gluon kernel. Do not edit.
# case_id=s2r-barrier-triton-data-hazard-01
# independent_label=DATA_HAZARD; actors=thread:0,thread:64; variant=1
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    ttgl.store(Data + lanes * 0 + 16, 102, mask=lanes == 0)
    value = ttgl.load(Data + lanes * 0 + 16, mask=lanes == 64, other=0)
    ttgl.store(Out + lanes * 0 + 1, value, mask=lanes == 64)
