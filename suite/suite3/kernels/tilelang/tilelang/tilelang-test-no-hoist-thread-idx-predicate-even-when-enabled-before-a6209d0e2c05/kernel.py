# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_loop_unswitching.py#L641-L651
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), dtype=T.int32)
        B = T.match_buffer(B_ptr, (256,), dtype=T.int32)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                for i in T.unroll(0, 2):
                    B[tx] = A[tx]
                    if tx == 0:
                        B[i] = T.int32(1)
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
