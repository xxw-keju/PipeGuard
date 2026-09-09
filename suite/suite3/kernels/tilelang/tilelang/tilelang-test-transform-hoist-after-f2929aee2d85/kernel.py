# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_hoist_broadcast_values.py#L74-L80
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            broadcast_var: T.float8_e4m3fn = T.float8_e4m3fn(1.2)
            broadcast_var_1: T.float8_e4m3fn = T.float8_e4m3fn(3.4)
            A_shared[0:8] = T.Broadcast(broadcast_var, 8) + T.Broadcast(broadcast_var_1, 8)
    return after


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
