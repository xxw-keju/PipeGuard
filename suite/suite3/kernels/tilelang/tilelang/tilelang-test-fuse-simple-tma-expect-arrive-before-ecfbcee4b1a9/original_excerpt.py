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
