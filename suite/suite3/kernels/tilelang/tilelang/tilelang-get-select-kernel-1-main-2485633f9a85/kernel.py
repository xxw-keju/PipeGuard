# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/language/test_tilelang_language_select.py#L8-L26
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        A: T.Tensor[(128, 8), T.float32],
        B: T.Tensor[(128, 8), T.float32],
    ):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding(0)
            tmp = T.alloc_var(T.bfloat16)
            tmp = A[tx, 3]

            B[tx, 0] = T.Select(True, A[tx, 0], 0.0)
            B[tx, 1] = T.Select(False, 1.0, A[tx, 1])
            B[tx, 2] = T.Select(T.cast(A[tx, 3], T.bfloat16) == tmp, A[tx, 2], T.cast(tmp, T.float32))
            B[tx, 3] = T.Select(B[tx, 0] != 0.0, T.if_then_else(B[tx, 1] != 0.0, A[tx, 3], 0.0), 0.0)

            for i in T.serial(4):
                B[tx, i + 4] = T.Select(
                    A[tx, 0] == 0, T.if_then_else(T.Select(True, False, True), 1.0, 2.0), T.Select(True, A[tx, i + 4], 3.0)
                )
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
