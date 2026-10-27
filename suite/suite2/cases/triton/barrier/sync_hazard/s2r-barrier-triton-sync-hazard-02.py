# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-barrier-triton-sync-hazard-02
# independent_label=SYNC_HAZARD; idiom=shared-memory handoff whose only ordering edge is a divergent barrier
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
    shared = ttgl.allocate_shared_memory(ttgl.int32, [128], _shared_layout())
    loaded = ttgl.load(Data + lanes, mask=lanes < 128, other=0)
    shared.store(loaded)
    token = ttgl.inline_asm_elementwise('.reg .pred p; setp.lt.u32 p, $1, 32; @p bar.sync 0; mov.u32 $0, $1;', '=r,r', [lanes], ttgl.int32, is_pure=False, pack=1)
    observed = shared.load(layout)
    ttgl.store(Out + lanes, observed + token * 0, mask=lanes == 0)
