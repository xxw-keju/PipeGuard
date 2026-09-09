# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/issue/test_tilelang_issue_3025.py#L25-L32
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

M = 4

N = 3

PITCH = 8

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        src: T.StridedTensor((M, N), (PITCH, 1), "int32"),
        out: T.Tensor((M,), "int32"),
    ):
        with T.Kernel(1, threads=1):
            for row in T.serial(M):
                out[row] = T.reinterpret(T.ldg32(src[row, 0]), "int32")
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
