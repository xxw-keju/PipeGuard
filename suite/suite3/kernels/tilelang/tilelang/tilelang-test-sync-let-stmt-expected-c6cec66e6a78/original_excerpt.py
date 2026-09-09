    @T.prim_func(private=True)
    def expected(A: T.Buffer((8192,), "float32")):
        blockIdx_x = T.launch_thread("blockIdx.x", 16)
        A_shared_1 = T.alloc_buffer((512,), "float32", scope="shared")
        in_thread_A_temp_1 = T.alloc_buffer((1,), "float32", scope="local")
        cross_thread_A_temp_1 = T.alloc_buffer((1,), "float32", scope="local")
        threadIdx_x = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        A_shared_1_1 = T.decl_buffer((512,), data=A_shared_1.data, scope="shared")
        for ax0 in range(512):
            A_shared_1_1[ax0] = A[blockIdx_x * 512 + ax0]
        in_thread_A_temp_1_1 = T.decl_buffer((1,), data=in_thread_A_temp_1.data, scope="local")
        in_thread_A_temp_1_1[0] = T.float32(0)
        T.tvm_storage_sync("shared")
        A_temp_0 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x])
        in_thread_A_temp_1_1[0] = A_temp_0
        A_temp_1 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 128])
        in_thread_A_temp_1_1[0] = A_temp_1
        A_temp_2 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 256])
        in_thread_A_temp_1_1[0] = A_temp_2
        A_temp_3 = T.bind(in_thread_A_temp_1_1[0] + A_shared_1_1[threadIdx_x + 384])
        in_thread_A_temp_1_1[0] = A_temp_3
        cross_thread_A_temp_1_1 = T.decl_buffer((1,), data=cross_thread_A_temp_1.data, scope="local")
        T.attr(
            T.comm_reducer(lambda x0, y0: x0 + y0, [T.float32(0)]),
            "reduce_scope",
            T.reinterpret(T.uint64(0), dtype="handle"),
        )
        T.tvm_thread_allreduce(
            T.uint32(1),
            in_thread_A_temp_1_1[0],
            T.bool(True),
            cross_thread_A_temp_1_1[0],
            threadIdx_x,
        )
