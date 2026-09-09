    @T.prim_func
    def after():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            counter = T.decl_buffer((1,), T.int32, scope="local")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            smem[0] = T.float16(0)
            counter[0] = 0
            T.fence_proxy_async()
            while counter[0] < T.int32(12):
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
                counter[0] = counter[0] + 1
