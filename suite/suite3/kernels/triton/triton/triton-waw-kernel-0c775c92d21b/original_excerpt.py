@triton.jit
def _waw_kernel(ptr, scratch_ptr, counter_ptr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(ptr, 1)
        tl.atomic_add(counter_ptr, 1, sem="relaxed")
    else:
        atomic_poll(counter_ptr, 1)
        tl.store(ptr, 2)
