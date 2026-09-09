# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_compile_errors.py#L526-L529
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.aggregate
class Square:
    x: tl.tensor

    @triton.constexpr_function
    def __init__(self, x):
        self.x = x

    @triton.must_use_result
    @triton.constexpr_function
    def power(self):
        return 2

    @triton.must_use_result
    @triton.jit
    def compute(self):
        return self.x * self.x

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def good_power_kernel():
        a = Square(tl.full((64, 64), 0.0, tl.float32))
        a = a.power()
    return good_power_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
