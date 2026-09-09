@triton.jit
def topk_kernel(K: tl.constexpr):
    x = tl.arange(0, 8)
    tl.topk(x, K)
