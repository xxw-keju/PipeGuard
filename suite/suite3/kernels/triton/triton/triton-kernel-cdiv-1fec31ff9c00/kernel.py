# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_line_info.py#L70-L74
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def kernel_cdiv(x):
    c = tl.full((32, 32), 4, dtype=tl.int8)
    d = tl.cdiv(c, 4)
    tl.device_print("", d)


def _resolve_kernel(**pipeguard_bindings):
    return kernel_cdiv


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
