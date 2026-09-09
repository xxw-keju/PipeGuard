# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_loop_unswitching.py#L314-L328
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def expected(
        A: T.Tensor((128,), T.float32),
        B: T.Tensor((128,), T.float32),
        C: T.Tensor((2,), T.float32),
    ):
        # Let bindings are hoisted before the if, redundant inner LetStmts are removed
        x = C[0]
        y = C[1]
        if x + y >= T.float32(0):
            for i in range(128):
                B[i] = A[i]
        else:
            for _i in range(128):
                T.evaluate(0)
    return expected


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
