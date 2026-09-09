# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L2016-L2032
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
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
    return serialized_add


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
