# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_compile_errors.py#L467-L470
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

extra_words = "These are extra words in the error message."

@triton.must_use_result(extra_words)
@triton.jit
def cube(x):
    return x * x * x

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def good_cube_kernel():
        a = tl.full((64, 64), 0.0, tl.float32)
        a = cube(a)
    return good_cube_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
