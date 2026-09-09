@triton.jit
def kernel_dot_combine(x):
    c = tl.full((32, 32), 4, dtype=tl.int8)
    a = (tl.arange(0, 32)[:, None] + tl.arange(0, 32)[None, :]).to(tl.int8)
    d = tl.dot(a, a)
    d = d + c
    tl.device_print("", d)
