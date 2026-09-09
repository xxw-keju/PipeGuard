# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L2610-L2620
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(X, Z1, Z2):
        offs = tl.arange(0, 2)
        x = tl.load(X + offs)
        z1, z2 = tl.split(x)
        tl.static_assert(isinstance(z1, tl.tensor))
        tl.static_assert(isinstance(z2, tl.tensor))
        tl.static_assert(z1.shape == [])
        tl.static_assert(z2.shape == [])
        tl.store(Z1, z1)
        tl.store(Z2, z2)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
