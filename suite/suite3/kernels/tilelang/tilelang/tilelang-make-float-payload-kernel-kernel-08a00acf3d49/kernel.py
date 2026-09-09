# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/tools/test_tilelang_tools_cuda_iket.py#L75-L79
# The exact upstream excerpt is stored in original_excerpt.py.

from __future__ import annotations

import tilelang.language as T

from tilelang.tools.cuda import iket

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            iket.mark("float_value", payload=iket.payload(A[0], dtype="float32"))
            A[0] = A[0]
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
