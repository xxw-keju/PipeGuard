# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L896-L913
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(dummy, N: tl.constexpr):
        offset1 = tl.arange(0, N)

        t = tl.unsqueeze(offset1, -1)
        tl.static_assert(t.shape == [N, 1])

        t = tl.unsqueeze(offset1, -2)
        tl.static_assert(t.shape == [1, N])

        t = tl.unsqueeze(tl.expand_dims(offset1, 0), -1)
        tl.static_assert(t.shape == [1, N, 1])

        t = tl.squeeze(tl.expand_dims(offset1, 1), -1)
        tl.static_assert(t.shape == [N])

        t = tl.squeeze(tl.expand_dims(offset1, 0), -2)
        tl.static_assert(t.shape == [N])
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
