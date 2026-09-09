# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_pipeline_planning.py#L347-L397
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def after(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        D: T.Tensor((64, 64), T.float16),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float16)

            for _i in T.serial(
                4,
                annotations={
                    "software_pipeline_async_producer_groups": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(-1),
                        T.int32(-1),
                        T.int32(-1),
                    ],
                    "software_pipeline_async_producers": [
                        T.int32(1),
                        T.int32(1),
                        T.int32(0),
                        T.int32(0),
                        T.int32(0),
                    ],
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [
                        T.int32(1),
                        T.int32(2),
                        T.int32(0),
                        T.int32(3),
                        T.int32(4),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(0),
                        T.int32(3),
                        T.int32(3),
                        T.int32(3),
                    ],
                    "tl_pipelined_num_stages": T.int32(3),
                },
            ):
                T.copy(A[0:64, 0:16], A_shared)
                T.copy(B[0:16, 0:64], B_shared)
                T.wgmma_gemm(A_shared, B_shared, C_local, clear_accum=True)
                T.wait_wgmma(0)
                T.copy(C_local, D[0:64, 0:64])
    return after


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
