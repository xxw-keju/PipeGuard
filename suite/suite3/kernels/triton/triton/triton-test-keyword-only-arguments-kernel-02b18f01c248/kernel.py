# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L5297-L5302
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def _keyword_only_impl(value=3, *, required, optional=5, REQUIRED: tl.constexpr, OPTIONAL: tl.constexpr = 2):
    return value + required + optional + REQUIRED * OPTIONAL

@triton.jit
def _keyword_only_varargs(value, *values, scale: tl.constexpr = 2):
    return value + values[0] + values[1] * scale

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(value=3, *, output, required, optional=5, REQUIRED: tl.constexpr, OPTIONAL: tl.constexpr = 2):
        tl.store(output, _keyword_only_impl(required=required, REQUIRED=REQUIRED))
        tl.store(output + 1,
                 _keyword_only_impl(value, required=required, optional=optional, REQUIRED=REQUIRED, OPTIONAL=OPTIONAL))
        tl.store(output + 2, _keyword_only_varargs(value, required, optional, scale=OPTIONAL))
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
