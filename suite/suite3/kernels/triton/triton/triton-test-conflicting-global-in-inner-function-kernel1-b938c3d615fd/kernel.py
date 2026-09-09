# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/runtime/test_cache.py#L483-L486
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

CONFLICTING_GLOBAL = tl.constexpr(0)

@triton.jit
def conflicting_global_inner():
    a = CONFLICTING_GLOBAL  # noqa

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel1():
        a = CONFLICTING_GLOBAL  # noqa
        conflicting_global_inner()
    return kernel1


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
