# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/print_helper.py#L89-L91
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

@triton.jit
def kernel_print_no_arg():
    print("no arg")


def _resolve_kernel(**pipeguard_bindings):
    return kernel_print_no_arg


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
