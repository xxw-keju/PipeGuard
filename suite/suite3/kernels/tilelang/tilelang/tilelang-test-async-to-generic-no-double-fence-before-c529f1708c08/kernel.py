# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_inject_fence_proxy.py#L27-L38
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            B_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            T.ptx_cp_async(
                T.tvm_access_ptr(T.type_annotation(T.uint8), A_shared.data, 0, 16, 2),
                T.tvm_access_ptr(T.type_annotation(T.uint8), B_shared.data, 0, 16, 1),
                16,
            )
            T.fence_proxy_async()
            T.call_extern("handle", "generic_op")
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
