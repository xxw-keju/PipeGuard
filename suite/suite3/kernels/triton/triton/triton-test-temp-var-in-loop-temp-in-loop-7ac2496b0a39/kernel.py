# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L6991-L7004
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def temp_in_loop(Z, N: tl.constexpr, BLOCK: tl.constexpr):
        acc = tl.full((BLOCK, ), 0, dtype=tl.int32)
        for i in range(N):
            if i == 0:
                temp = tl.full((BLOCK, ), 2, dtype=tl.int32)
                acc = temp
            else:
                acc += tl.full((BLOCK, ), 1, dtype=tl.int32)
            # reuse the temp variable and make sure to check that it isn't creating incorrect IR.
            temp = tl.full((BLOCK, ), 1, dtype=tl.int32)
            acc += temp
        z = Z + tl.arange(0, BLOCK)
        tl.store(z, acc)
    return temp_in_loop


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
