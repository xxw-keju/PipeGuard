# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L3673-L3678
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(In, Out, in_shape1: tl.constexpr, in_shape2: tl.constexpr, ou_shape1: tl.constexpr,
               ou_shape2: tl.constexpr, trans1: tl.constexpr, trans2: tl.constexpr):
        in_offs = tl.arange(0, in_shape1)[:, None] * in_shape2 + tl.arange(0, in_shape2)[None, :]
        ou_offs = tl.arange(0, ou_shape1)[:, None] * ou_shape2 + tl.arange(0, ou_shape2)[None, :]
        tl.store(Out + ou_offs, tl.permute(tl.load(In + in_offs), (trans1, trans2)))
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
