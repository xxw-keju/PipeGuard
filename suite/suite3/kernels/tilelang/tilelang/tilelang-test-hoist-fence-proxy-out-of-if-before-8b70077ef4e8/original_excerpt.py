    @T.prim_func
    def before(flag: T.int32):
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            desc_a = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            desc_b = T.decl_buffer((1,), T.uint64, scope="local.descriptor.wgmma")
            C_local = T.decl_buffer((32,), T.float16, scope="local")
            smem[0] = T.float16(0)
            T.warpgroup_arrive()
            if flag == 1:
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
            else:
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
