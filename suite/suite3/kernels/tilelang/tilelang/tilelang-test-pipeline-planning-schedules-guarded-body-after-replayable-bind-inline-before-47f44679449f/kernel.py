# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_pipeline_planning.py#L191-L204
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(
        KV: T.Tensor((4, 4), T.bfloat16),
        ids: T.Tensor((4,), T.int32),
        C: T.Tensor((4,), T.bfloat16),
    ):
        with T.Kernel(4, threads=1):
            A = T.alloc_shared((4,), T.bfloat16)

            for i in T.Pipelined(4, num_stages=2):
                if i > 1:
                    idx = ids[i]
                    T.copy(KV[idx, :], A)
                    T.copy(A, C)
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
