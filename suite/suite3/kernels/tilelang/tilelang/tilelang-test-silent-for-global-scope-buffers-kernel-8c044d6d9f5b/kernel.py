# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_verify_buffer_init.py#L114-L118
# The exact upstream excerpt is stored in original_excerpt.py.

import torch

import tilelang.language as T

BC, BK, NB = 16, 64, 16

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((NB, BC), torch.float32), C: T.Tensor((NB, BC), torch.float32)):
        with T.Kernel(NB, threads=64) as bx:
            for i in T.Parallel(BC):
                C[bx, i] = A[bx, i]
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
