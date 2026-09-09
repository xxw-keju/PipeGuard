    @T.prim_func
    def before_no_set_max_nreg(A: T.Tensor((512, 512), T.float16)):
        bx = T.launch_thread("blockIdx.x", 8)
        v = T.launch_thread("threadIdx.x", 128)

        with T.sblock(""):
            T.reads(A[bx * 64, 0:64])
            T.writes()

            # Add no_set_max_nreg to disable register hinting
            T.disable_warp_group_reg_alloc()

            mbars = T.alloc_barrier([128, 128])  # noqa: F841
            T.attr([128, 128], "kWarpSpecializationScope", 0)

            if v >= 128:
                # Producer branch - should not have set_max_nreg calls
                T.evaluate(0)
            else:
                # Consumer branch - should not have set_max_nreg calls
                T.evaluate(0)
