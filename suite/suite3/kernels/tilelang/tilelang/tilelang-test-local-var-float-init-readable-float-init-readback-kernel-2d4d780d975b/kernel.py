# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/amd/test_tilelang_hip_codegen.py#L522-L527
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def kernel(out: T.Tensor((1,), T.float32)) -> None:
        with T.Kernel(1, threads=32) as _:
            v = T.alloc_var(T.float32, init=3.14)
            if T.get_thread_binding() == 0:
                out[0] = v
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
