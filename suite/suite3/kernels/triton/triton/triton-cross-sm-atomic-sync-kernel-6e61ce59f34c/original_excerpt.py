@triton.jit
def _cross_sm_atomic_sync_kernel(payload_ptr, flag_ptr, out_ptr, producer_sem: tl.constexpr, consumer_sem: tl.constexpr,
                                 scope: tl.constexpr):
    pid = tl.program_id(0)
    if pid == 0:
        tl.store(payload_ptr, 1000)
        tl.atomic_xchg(flag_ptr, 1, sem=producer_sem, scope=scope)
    elif pid == 1:
        atomic_poll(flag_ptr, 1, sem=consumer_sem, scope=scope)
        result = tl.load(payload_ptr)
        tl.store(out_ptr, result)
