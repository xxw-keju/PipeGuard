# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-barrier-triton-safe-04
# independent_label=SAFE; idiom=two-way request/response handoff: two full CTA barrier rounds order a global write, a cross-warp reply write, and the final read-back
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    ttgl.store(Data + lanes * 0 + 8, 21, mask=lanes == 0)
    ttgl.barrier()
    request = ttgl.load(Data + lanes * 0 + 8, mask=lanes == 64, other=0)
    ttgl.store(Data + lanes * 0 + 12, request + 1, mask=lanes == 64)
    ttgl.barrier()
    reply = ttgl.load(Data + lanes * 0 + 12, mask=lanes == 0, other=0)
    ttgl.store(Out + lanes * 0 + 4, reply, mask=lanes == 0)
