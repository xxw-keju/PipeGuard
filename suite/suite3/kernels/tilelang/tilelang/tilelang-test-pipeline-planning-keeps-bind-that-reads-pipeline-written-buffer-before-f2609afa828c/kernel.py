# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_pipeline_planning.py#L789-L800
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(
        A: T.Tensor((64,), T.float16),
        C: T.Tensor((64,), T.float16),
    ):
        with T.Kernel(1, threads=16):
            A_shared = T.alloc_shared((16,), T.float16)
            for i in T.Pipelined(4, order=[0, 1, 2], stage=[0, 1, 1]):
                base: T.int32 = i * 16
                T.copy(A[base], A_shared)
                value: T.float16 = A_shared[0]
                C[base] = value
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
