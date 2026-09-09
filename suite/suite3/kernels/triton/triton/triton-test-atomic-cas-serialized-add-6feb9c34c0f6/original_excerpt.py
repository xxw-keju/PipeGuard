    @triton.jit
    def serialized_add(data, Lock, triton_dtype: tl.constexpr, SEM: tl.constexpr):
        num0 = tl.full((1, ), 0, dtype=triton_dtype).item()
        num1 = tl.full((1, ), 1, dtype=triton_dtype).item()

        ptrs = data + tl.arange(0, 128)
        while tl.atomic_cas(Lock, num0, num1, SEM) == 1:
            pass

        tl.store(ptrs, tl.load(ptrs) + 1.0)

        # insert barrier to set a fence between tl.store and
        # tl.atomic_xchg in a block.
        tl.debug_barrier()

        # release lock
        tl.atomic_xchg(Lock, num0)
