    @T.prim_func
    def before(A: T.Tensor((40,), T.uint8), B: T.Tensor((40,), T.uint8)):
        S = T.alloc_buffer((4,), dtype=T.uint8, scope="shared")
        for i in T.serial(
            0,
            5,
            annotations={
                "software_pipeline_stage": [0, 3],
                "software_pipeline_order": [0, 1],
                "software_pipeline_async_stages": [0],
                "software_pipeline_async_producers": [1, 0],
                "software_pipeline_async_producer_groups": [0, -1],
            },
        ):
            with T.sblock("copy"):
                T.reads(A[i * 4 : i * 4 + 4])
                T.writes(S[0:4])
                T.copy(A[i * 4 : i * 4 + 4], S[0:4])
            with T.sblock("consume"):
                T.reads(S[0:4])
                T.writes(B[i * 4 : i * 4 + 4])
                for j in range(4):
                    B[i * 4 + j] = S[j]
