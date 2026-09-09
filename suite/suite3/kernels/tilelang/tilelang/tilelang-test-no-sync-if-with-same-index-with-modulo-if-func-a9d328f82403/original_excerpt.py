    @T.prim_func(check_well_formed=False)
    def func() -> None:
        threadIdx_x = T.env_thread("threadIdx.x")
        blockIdx_x = T.env_thread("blockIdx.x")
        p0 = T.alloc_buffer([1], dtype="float32", scope="local")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        temp_shared = T.alloc_buffer([32], dtype="float32", scope="shared")
        T.launch_thread(blockIdx_x, 1)
        T.launch_thread(threadIdx_x, 32)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        if threadIdx_x % 4 == 0:
            temp_shared[threadIdx_x] = p0[0]
        result_local[0] = temp_shared[threadIdx_x]
