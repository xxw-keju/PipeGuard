# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/print_helper.py#L50-L57
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def kernel_device_print_large(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x = tl.full([BLOCK_M, BLOCK_N], 1, tl.int32)
    # Triton should change this prefix to "x: ".
    tl.device_print("x ", x)


def _resolve_kernel(**pipeguard_bindings):
    return kernel_device_print_large


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
