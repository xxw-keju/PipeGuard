# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_pipeline_planning.py#L133-L151
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.serial(
            4,
            annotations={
                "software_pipeline_async_producer_groups": [
                    T.int32(0),
                    T.int32(-1),
                ],
                "software_pipeline_async_producers": [T.int32(1), T.int32(0)],
                "software_pipeline_async_stages": [T.int32(0)],
                "software_pipeline_order": [T.int32(0), T.int32(1)],
                "software_pipeline_stage": [T.int32(0), T.int32(1)],
                "tl_pipelined_num_stages": T.int32(2),
            },
        ):
            T.copy(A[i * 16], S)
            T.copy(S, C[i * 16])
    return after


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
