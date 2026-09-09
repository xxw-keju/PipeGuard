    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            A_shared = T.alloc_shared((64, 64), T.float16)
            A_frag = T.alloc_fragment((64, 64), T.float16)
            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("Producer", warps_lo=0, warps_hi=1, max_nreg=40),
                        T.WSRole("Consumer", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[T.WSPipeline("buf", [A_shared], depth=2)],
                    scopes=[
                        T.WSScope(
                            "loop_wave",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "copy_frag",
                                    T.WSSync.consumer_release("buf"),
                                    "copy_out",
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"Producer": ["loop_wave"], "Consumer": ["loop_wave"]},
                        ),
                    ],
                )
            )
            with T.ws_op("loop_wave"):
                while A_shared[0, 0] > T.float16(0):
                    T.copy(A[0, 0], A_shared, annotations={T.WSID: "copy_in"})
                    T.copy(A_shared, A_frag, annotations={T.WSID: "copy_frag"})
                    T.copy(A_frag, B[0, 0], annotations={T.WSID: "copy_out"})
