# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/tile-ai/tilelang/blob/2c31e071c626c0498b48dcdc2741dc8651b70dc1/testing/python/transform/test_tilelang_transform_inject_fence_proxy.py#L387-L440
# The exact upstream excerpt is stored in original_excerpt.py.

import tilelang.language as T

def _resolve_kernel(**pipeguard_bindings):
    @T.prim_func
    def before(A_desc: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.uint8, scope="shared.dyn")
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")

            # Local stores should not be treated as generic proxy traffic.
            C_local[0] = T.float16(0)

            # Descriptor initialization is metadata only and should not be
            # treated as generic proxy traffic.
            T.initialize_wgmma_descriptor(desc_a, T.uint64(0), 2, 1, 32)
            T.initialize_wgmma_descriptor(desc_b, T.uint64(0), 2, 1, 32)

            if T.shuffle_elect(0):
                T.ptx_init_barrier_thread_count(mbarrier[0], 128)
            T.ptx_fence_barrier_init()
            T.tvm_storage_sync("shared")

            # Barrier ops should not be classified as generic proxy traffic.
            T.mbarrier_wait_parity(mbarrier[0], 0)
            T.mbarrier_expect_tx(mbarrier[0], 16)
            T.tma_load(
                A_desc,
                mbarrier[0],
                T.tvm_access_ptr(T.type_annotation(T.uint8), smem.data, 0, 16, 2),
                0,
                0,
                0,
            )

            # Another async proxy op after barrier ops.
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
