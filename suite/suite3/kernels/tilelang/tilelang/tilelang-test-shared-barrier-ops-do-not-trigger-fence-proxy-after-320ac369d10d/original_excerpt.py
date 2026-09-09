    @T.prim_func
    def after(A_desc: T.handle("uint8x128", "grid_constant")):
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.uint8, scope="shared.dyn")
            mbarrier = T.decl_buffer((1,), T.uint64, scope="shared.barrier")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")

            C_local[0] = T.float16(0)

            T.initialize_wgmma_descriptor(desc_a, T.uint64(0), 2, 1, 32)
            T.initialize_wgmma_descriptor(desc_b, T.uint64(0), 2, 1, 32)

            if T.shuffle_elect(0):
                T.ptx_init_barrier_thread_count(mbarrier[0], 128)
            T.ptx_fence_barrier_init()
            T.tvm_storage_sync("shared")

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
