# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-memory-triton-sync-hazard-02
# independent_label=SYNC_HAZARD; idiom=WAR guarded only by a divergent CTA barrier (96 of 128 lanes arrive)
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
    token = ttgl.inline_asm_elementwise('.reg .pred p; setp.lt.u32 p, $1, 96; @p bar.sync 0; mov.u32 $0, $1;', '=r,r', [lanes], ttgl.int32, is_pure=False, pack=1)
    ttgl.store(Data + lanes * 0 + 24, 3 + token * 0, mask=lanes == 64)
    ttgl.store(Out + lanes * 0 + 1, value, mask=lanes == 0)
