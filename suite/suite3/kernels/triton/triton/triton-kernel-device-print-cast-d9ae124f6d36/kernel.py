# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/print_helper.py#L22-L25
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def kernel_device_print_cast(BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK) + 128
    tl.device_print("x: ", x.to(tl.uint8))


def _resolve_kernel(**pipeguard_bindings):
    return kernel_device_print_cast


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
