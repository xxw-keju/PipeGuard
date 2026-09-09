# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_verify_buffer_init.py#L399-L409
# The exact upstream excerpt is stored in original_excerpt.py.

import torch

import tilelang.language as T

BC, BK, NB = 16, 64, 16

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((BC, BK), torch.float32), B: T.Tensor((BC,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            a_f = T.alloc_fragment((BC, BK), torch.float32)
            r_f = T.alloc_reducer((BC,), torch.float32, op="sum", replication="all")
            T.clear(r_f)
            T.copy(A, a_f)
            for i, j in T.Parallel(BC, BK):
                r_f[i] += a_f[i, j]
            T.finalize_reducer(r_f)
            T.copy(r_f, B)
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
