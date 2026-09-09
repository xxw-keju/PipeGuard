# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L7521-L7530
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(X, Y, Z, RANK: tl.constexpr, TRANS_A: tl.constexpr, TRANS_B: tl.constexpr):
        x = tl.load(X + tl.arange(0, 256 << RANK)).reshape([2] * (RANK - 2) + [32, 32])
        y = tl.load(Y + tl.arange(0, 256 << RANK)).reshape([2] * (RANK - 2) + [32, 32])
        if TRANS_A:
            x = tl.trans(x)
        if TRANS_B:
            y = tl.trans(y)
        z = tl.dot(x, y)
        tl.store(Z + tl.arange(0, 256 << RANK), z.reshape([256 << RANK]))
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
