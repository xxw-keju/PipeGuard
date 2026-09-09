# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_tuple.py#L394-L396
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

from typing import NamedTuple

@triton.constexpr_function
def passthrough_constexpr(x):
    return x

class TrivialTuple(NamedTuple):
    foo: tl.constexpr

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel():
        tl.static_assert(passthrough_constexpr(TrivialTuple(0)).foo == 0)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
