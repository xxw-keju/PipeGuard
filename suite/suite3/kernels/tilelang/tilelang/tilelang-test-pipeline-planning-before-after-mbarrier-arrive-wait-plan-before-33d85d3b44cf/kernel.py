# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_pipeline_planning.py#L217-L226
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((16,), T.float16)
            mbar = T.alloc_barrier(1)
            for i in T.Pipelined(4, num_stages=2):
                T.copy(A[i * 16], S)
                T.mbarrier_arrive(mbar)
                T.mbarrier_wait_parity(mbar, i % 2)
                T.copy(S, C[i * 16])
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
