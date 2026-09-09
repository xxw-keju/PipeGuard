# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_tuple.py#L375-L377
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def _namedtuple_float_tuple_kernel():
        x, y = float("-inf"), float("inf")  # noqa: F841
    return _namedtuple_float_tuple_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
