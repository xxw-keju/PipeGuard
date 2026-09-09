# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_verify_buffer_init.py#L423-L435
# The exact upstream excerpt is stored in original_excerpt.py.

import torch

import tilelang.language as T

BC, BK, NB = 16, 64, 16

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(A: T.Tensor((BC,), torch.float32), B: T.Tensor((1,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            src = T.alloc_fragment((BC,), torch.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), torch.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(BC):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), torch.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
