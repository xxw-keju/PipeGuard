# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_codegen_line_directives.py#L28-L32
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

N = 128

@T.prim_func
def vec_add_cuda(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=N) as bx:
        for i in T.Parallel(N):
            B[bx * N + i] = A[bx * N + i] + 1.0  # line_marker_store_cuda


def _resolve_kernel(**pipeguard_bindings):
    return vec_add_cuda


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
