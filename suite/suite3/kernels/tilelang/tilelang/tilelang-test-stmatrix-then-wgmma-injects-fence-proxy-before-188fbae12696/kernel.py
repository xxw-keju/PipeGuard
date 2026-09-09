# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_inject_fence_proxy.py#L883-L916
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

from tvm import tirx

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.call_intrin(
                "handle",
                tirx.op.Op.get("tl.ptx_stmatrix"),
                T.int32(0),
                1,
                T.tvm_access_ptr(T.type_annotation(T.float16), smem.data, 0, 16, 2),
                T.int32(0),
            )
            T.warpgroup_arrive()
            T.ptx_wgmma_ss(
                T.float16,
                "m64n64k16",
                T.bool(True),
                T.bool(True),
                "fp16",
                "fp16",
                "fp16",
                desc_a.data,
                T.int32(0),
                desc_b.data,
                T.int32(0),
                C_local.data,
                T.int32(0),
                T.bool(True),
                1,
                1,
            )
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
