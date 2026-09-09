# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_unroll.py#L8-L15
# The exact upstream excerpt is stored in original_excerpt.py.

from tilelang import language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype=T.float32, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                for i in T.unroll(0, 16, step=4):
                    A[0, i] = 1.0
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
