# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_fuse_mbarrier_arrive_expect_tx.py#L32-L47
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(A_desc: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1):
            smem = T.decl_buffer((16,), T.uint8, scope="shared.dyn")
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            if T.shuffle_elect(0):
                T.mbarrier_expect_tx(mbarrier[0], 16)
                T.tma_load(
                    A_desc,
                    mbarrier[0],
                    T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 0, 16, 2),
                    0,
                    0,
                    0,
                )
                T.ptx_arrive_barrier(mbarrier[0])
    return before


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
