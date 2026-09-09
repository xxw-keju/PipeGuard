# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/cpu/test_tilelang_cpu_fp16_const.py#L56-L60
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(A: T.Tensor((4,), "float16"), B: T.Tensor((4,), "float16")):
        with T.Kernel(1):
            for i in T.grid(4):
                B[i] = A[i] + T.float16(1.5) + T.float16(2.25)
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
