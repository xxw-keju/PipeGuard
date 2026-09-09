@triton.jit
def kernel_device_print_cast(BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK) + 128
    tl.device_print("x: ", x.to(tl.uint8))
