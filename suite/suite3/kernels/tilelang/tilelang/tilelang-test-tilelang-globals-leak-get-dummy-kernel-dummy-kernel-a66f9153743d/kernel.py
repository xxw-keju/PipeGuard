# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_memory_leak.py#L15-L26
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def dummy_kernel(
        a: T.Tensor[(1,), T.float32],
    ):
        with T.Kernel(1) as _:
            # `ub = T.alloc_shared(...)` is an assignment that goes through
            # `__tb.bind` -> `get_parent_locals`. Without it the eager builder
            # never calls `get_parent_locals` (subscript stores and `with ... as`
            # targets do not), so this test cannot detect the frame leak.
            ub = T.alloc_shared((1,), "float32")
            T.copy(a, ub)
            T.copy(ub, a)
    return dummy_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
