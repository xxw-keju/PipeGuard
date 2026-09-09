    @T.prim_func
    def after(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        with T.Kernel(1, threads=1):
            S = T.alloc_shared((16,), T.float16)
            mbar = T.alloc_barrier(1)
            for i in T.serial(
                4,
                annotations={
                    "software_pipeline_async_producer_groups": [
                        T.int32(0),
                        T.int32(-1),
                        T.int32(-1),
                        T.int32(-1),
                    ],
                    "software_pipeline_async_producers": [
                        T.int32(1),
                        T.int32(0),
                        T.int32(0),
                        T.int32(0),
                    ],
                    "software_pipeline_async_stages": [T.int32(0)],
                    "software_pipeline_order": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(2),
                        T.int32(3),
                    ],
                    "software_pipeline_stage": [
                        T.int32(0),
                        T.int32(1),
                        T.int32(1),
                        T.int32(1),
                    ],
                    "tl_pipelined_num_stages": T.int32(2),
                },
            ):
                T.tma_copy(A[i * 16], S, barrier=mbar)
                T.barrier_arrive(mbar)
                T.mbarrier_wait_parity(mbar, i % 2)
                T.copy(S, C[i * 16])
