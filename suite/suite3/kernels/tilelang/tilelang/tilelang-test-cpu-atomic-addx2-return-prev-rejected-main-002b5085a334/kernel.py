# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/cpu/test_tilelang_cpu_atomic.py#L378-L385
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        Dst: T.Tensor((2,), "float16"),
        Val: T.Tensor((2,), "float16"),
        Prev: T.Tensor((2,), "float16"),
    ):
        with T.Kernel(1):
            Prev[0:2] = T.atomic_addx2(Dst[0:2], Val[0:2], return_prev=True)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
