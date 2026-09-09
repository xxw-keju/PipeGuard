# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/cpu/test_tilelang_cpu_vec_type.py#L26-L42
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

N = 256

@T.prim_func
def vec_arith(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    Add: T.Tensor((N,), "float32"),
    Sub: T.Tensor((N,), "float32"),
    Mul: T.Tensor((N,), "float32"),
    Div: T.Tensor((N,), "float32"),
):
    for i in T.Parallel(N):
        Add[i] = A[i] + B[i]
    for i in T.Parallel(N):
        Sub[i] = A[i] - B[i]
    for i in T.Parallel(N):
        Mul[i] = A[i] * B[i]
    for i in T.Parallel(N):
        Div[i] = A[i] / B[i]


def _resolve_kernel(**pipeguard_bindings):
    return vec_arith


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
