# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_line_info.py#L39-L41
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit(noinline=True)
def device_noinline(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = x + x
    tl.store(Y + tl.arange(0, BLOCK), y)

@triton.jit
def kernel_call_noinline(X, Y, BLOCK: tl.constexpr):
    device_noinline(X, Y, BLOCK)


def _resolve_kernel(**pipeguard_bindings):
    return kernel_call_noinline


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
