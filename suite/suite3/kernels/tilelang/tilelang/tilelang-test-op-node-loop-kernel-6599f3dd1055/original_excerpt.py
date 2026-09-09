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
                            "loop_k",
                            {
                                "Producer": [
                                    T.WSSync.producer_acquire("buf"),
                                    "copy_in",
                                    T.WSSync.producer_commit("buf"),
                                ],
                                "Consumer": [
                                    T.WSSync.consumer_wait("buf"),
                                    "store_loop",
                                    T.WSSync.consumer_release("buf"),
                                ],
                            },
                        ),
                        T.WSScope(T.WSScope.ROOT, {"Producer": ["loop_k"], "Consumer": ["loop_k"]}),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=2, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], A_shared, annotations={T.WSID: "copy_in"})
                # The unrolled loop is one op node; its statements are
                # unannotated and its A_shared read binds it to the pipeline.
                for s in T.unroll(2, annotations={T.WSID: "store_loop"}):
                    T.copy(A_shared[:, s * 32 : (s + 1) * 32], A_frag[:, s * 32 : (s + 1) * 32])
                    T.copy(A_frag[:, s * 32 : (s + 1) * 32], B[i * 64, s * 32])
