@triton.autotune(
    configs=[
        triton.Config({"BLOCK": 64}, num_warps=4),
    ],
    key=[],
)
@triton.jit
def kernel_autotune(X, Y, SIZE: tl.constexpr, BLOCK: tl.constexpr):
    for i in range(0, SIZE, BLOCK):
        x = tl.load(X + i + tl.arange(0, BLOCK))
        tl.store(Y + i + tl.arange(0, BLOCK), x)
