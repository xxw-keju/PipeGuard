# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/issue/test_tilelang_issue_2682.py#L18-L26
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

NUM_ELEMENTS = 4

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(
        source: T.Tensor[(NUM_ELEMENTS,), T.float32],
        destination: T.Tensor[(NUM_ELEMENTS,), T.float32],
    ):
        with T.Kernel(1, threads=1):
            for index in T.vectorized(NUM_ELEMENTS):
                value = source[index]
                destination[index] = T.Select(value > 0.0, value, value + 1.0)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
