# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/test_debug.py#L86-L88
# The exact upstream excerpt is stored in original_excerpt.py.

import triton.language as tl

import triton

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def _kernel(COND: tl.constexpr):
        tl.static_assert(COND)
    return _kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
