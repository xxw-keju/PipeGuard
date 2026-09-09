    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((256,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            T.evaluate(T.call_extern("handle", "custom_ptr_store", T.address_of(smem[0])))
            T.warpgroup_arrive()
            T.fence_proxy_async()
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
