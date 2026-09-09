# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_readonly_param_const_codegen.py#L8-L15
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def main(
        x: T.Tensor((128,), T.float32),
        y: T.Tensor((128,), T.float32),
    ):
        # One-dimensional kernel; writes y from x without modifying x
        with T.Kernel(128, threads=32) as pid:
            y[pid] = x[pid] + 1.0
    return main


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
