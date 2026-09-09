# Generated Recovery Suite 2 real Triton Gluon kernel. Do not edit.
# case_id=s2r-barrier-triton-sync-hazard-04
# independent_label=SYNC_HAZARD; actors=thread:0,thread:64; variant=4
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    token = ttgl.inline_asm_elementwise(
        """.reg .pred p; setp.lt.u32 p, $1, 96; @p bar.sync 0; mov.u32 $0, $1;""",
        "=r,r", [lanes], ttgl.int32, is_pure=False, pack=1,
    )
    ttgl.store(Out + lanes * 0 + 4, token, mask=lanes == 0)
