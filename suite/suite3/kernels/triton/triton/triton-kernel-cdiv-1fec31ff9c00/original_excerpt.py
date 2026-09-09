@triton.jit
def kernel_cdiv(x):
    c = tl.full((32, 32), 4, dtype=tl.int8)
    d = tl.cdiv(c, 4)
    tl.device_print("", d)
