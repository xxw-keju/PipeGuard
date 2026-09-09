@triton.jit
def scalar_tensor_kernel(out, value_scalar):
    tl.store(out, value_scalar)
