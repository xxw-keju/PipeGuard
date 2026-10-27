# Redesigned Suite 2 real Triton Gluon kernel (structural-diversity rework). Do not hand edit.
# case_id=s2r-barrier-triton-sync-hazard-04
# independent_label=SYNC_HAZARD; idiom=three-round handshake ring whose middle rendezvous is divergent (full -> divergent -> full)
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
    ttgl.barrier()
    shared.store(ttgl.full([128], 11, ttgl.int32, layout))
    token = ttgl.inline_asm_elementwise('.reg .pred p; setp.lt.u32 p, $1, 64; @p bar.sync 0; mov.u32 $0, $1;', '=r,r', [lanes], ttgl.int32, is_pure=False, pack=1)
    observed = shared.load(layout)
    ttgl.barrier()
    ttgl.store(Out + lanes * 0 + 4, observed + token * 0, mask=lanes == 64)
