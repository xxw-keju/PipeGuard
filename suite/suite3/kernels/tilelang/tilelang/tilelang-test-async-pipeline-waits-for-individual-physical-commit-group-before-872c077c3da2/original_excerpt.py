    @T.prim_func
    def before(
        A: T.Tensor((16, 16), T.float32),
        B: T.Tensor((16, 16), T.float32),
        D: T.Tensor((16, 16), T.float32),
        C: T.Tensor((16, 16), T.float32),
    ):
        for tx in T.thread_binding(0, 16, thread="threadIdx.x"):
            A_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            B_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            D_shared = T.alloc_buffer((16, 1), dtype=T.float32, scope="shared")
            for i in T.serial(
                0,
                8,
                annotations={
                    "software_pipeline_stage": [0, 3, 0, 3, 0, 3],
                    "software_pipeline_order": [1, 0, 3, 2, 5, 4],
                    "software_pipeline_async_stages": [0],
                    "software_pipeline_async_producers": [1, 0, 1, 0, 1, 0],
                    "software_pipeline_async_producer_groups": [0, -1, 1, -1, 2, -1],
                    "tl_pipelined_num_stages": 3,
                },
            ):
                with T.sblock("copy_a"):
                    T.reads(A[tx, i])
                    T.writes(A_shared[tx, 0])
                    A_shared[tx, 0] = A[tx, i]
                with T.sblock("consume_a"):
                    T.reads(A_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = A_shared[tx, 0]
                with T.sblock("copy_b"):
                    T.reads(B[tx, i])
                    T.writes(B_shared[tx, 0])
                    B_shared[tx, 0] = B[tx, i]
                with T.sblock("consume_b"):
                    T.reads(B_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = C[tx, i] + B_shared[tx, 0]
                with T.sblock("copy_d"):
                    T.reads(D[tx, i])
                    T.writes(D_shared[tx, 0])
                    D_shared[tx, 0] = D[tx, i]
                with T.sblock("consume_d"):
                    T.reads(D_shared[tx, 0])
                    T.writes(C[tx, i])
                    C[tx, i] = C[tx, i] + D_shared[tx, 0]
