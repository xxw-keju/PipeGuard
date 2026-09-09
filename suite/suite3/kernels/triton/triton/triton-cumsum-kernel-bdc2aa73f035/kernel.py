# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/runtime/test_compilation_listener.py#L13-L17
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def cumsum_kernel(ptr):
    block = ptr + tl.arange(0, 4)
    x = tl.load(block)
    tl.store(block, tl.cumsum(x, 0))


def _resolve_kernel(**pipeguard_bindings):
    return cumsum_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
