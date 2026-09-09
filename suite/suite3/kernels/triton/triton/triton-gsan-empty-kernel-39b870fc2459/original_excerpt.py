@triton.jit
def _gsan_empty_kernel(out_ptr):
    tl.store(out_ptr, 0)
