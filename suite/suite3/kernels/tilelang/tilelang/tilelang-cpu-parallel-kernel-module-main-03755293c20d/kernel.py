# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_lower_tile_op.py#L394-L401
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A: T.Tensor((16,), "int32"), B: T.Tensor((16,), "int32")):
        with T.Kernel(1):
            frag = T.alloc_fragment((16,), "int32")
            for i in T.Parallel(16):
                frag[i] = A[i]
            for i in T.Parallel(16):
                B[i] = frag[i]
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
