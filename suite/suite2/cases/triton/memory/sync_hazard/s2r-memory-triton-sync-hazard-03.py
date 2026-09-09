# Generated Recovery Suite 2 real Triton Gluon kernel. Do not edit.
# case_id=s2r-memory-triton-sync-hazard-03
# independent_label=SYNC_HAZARD; actors=thread:0,thread:64; variant=3
from triton.experimental import gluon
from triton.experimental.gluon import language as ttgl

@gluon.constexpr_function
def _layout():
    return ttgl.BlockedLayout(size_per_thread=[1], threads_per_warp=[32], warps_per_cta=[4], order=[0], cga_layout=[])

@gluon.jit
def kernel(Data, Out):
    layout: ttgl.constexpr = _layout()
    lanes = ttgl.arange(0, 128, layout)
    ttgl.store(Data + lanes * 0 + 32, 104, mask=lanes == 0)
    token = ttgl.inline_asm_elementwise(
        """.reg .pred p; setp.lt.u32 p, $1, 64; @p bar.sync 0; mov.u32 $0, $1;""",
        "=r,r", [lanes], ttgl.int32, is_pure=False, pack=1,
    )
    value = ttgl.load(Data + lanes * 0 + 32, mask=lanes == 64, other=0)
    ttgl.store(Out + lanes * 0 + 3, value, mask=lanes == 64)
