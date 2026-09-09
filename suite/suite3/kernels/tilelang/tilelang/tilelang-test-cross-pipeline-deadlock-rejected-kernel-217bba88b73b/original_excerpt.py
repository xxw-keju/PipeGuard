    @T.prim_func
    def kernel(A: T.Tensor((256, 64), T.float16), B: T.Tensor((256, 64), T.float16)):
        with T.Kernel(1, threads=128) as _:
            X_shared = T.alloc_shared((64, 64), T.float16)
            Y_shared = T.alloc_shared((64, 64), T.float16)
            X_frag = T.alloc_fragment((64, 64), T.float16)

            T.annotate_ws_schedule(
                T.WSSchedule(
                    num_warps=8,
                    roles=[
                        T.WSRole("RoleX", warps_lo=0, warps_hi=4, max_nreg=224),
                        T.WSRole("RoleY", warps_lo=4, warps_hi=8, max_nreg=224),
                    ],
                    pipelines=[
                        T.WSPipeline("x", [X_shared], depth=1),
                        T.WSPipeline("y", [Y_shared], depth=1),
                    ],
                    scopes=[
                        T.WSScope(
                            "loop_k",
                            {
                                # RoleX produces x but first waits on y;
                                # RoleY produces y but first waits on x.
                                "RoleX": [
                                    T.WSSync.consumer_wait("y"),
                                    T.WSSync.producer_acquire("x"),
                                    "copy_x",
                                    T.WSSync.producer_commit("x"),
                                    "read_y",
                                    T.WSSync.consumer_release("y"),
                                    "copy_out_x",
                                ],
                                "RoleY": [
                                    T.WSSync.consumer_wait("x"),
                                    T.WSSync.producer_acquire("y"),
                                    "copy_y",
                                    T.WSSync.producer_commit("y"),
                                    T.WSSync.consumer_release("x"),
                                ],
                            },
                        ),
                        T.WSScope(
                            T.WSScope.ROOT,
                            {"RoleX": ["loop_k"], "RoleY": ["loop_k"]},
                        ),
                    ],
                )
            )

            for i in T.Pipelined(4, num_stages=1, annotations={T.WSID: "loop_k"}):
                T.copy(A[i * 64, 0], X_shared, annotations={T.WSID: "copy_x"})
                T.copy(X_shared, Y_shared, annotations={T.WSID: "copy_y"})
                T.copy(Y_shared, X_frag, annotations={T.WSID: "read_y"})
                T.copy(X_frag, B[i * 64, 0], annotations={T.WSID: "copy_out_x"})
