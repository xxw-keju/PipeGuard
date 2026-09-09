    @T.prim_func(private=True)
    def func():
        token_ids = T.alloc_buffer([128], dtype="int32", scope="shared")
        data_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        # First phase: all threads write to data_shared
        data_shared[tx] = T.float32(tx)
        # Non-uniform condition: reads shared memory with threadIdx-dependent index
        # token_ids[tx] can be different for each thread (e.g., some are -1, some are valid)
        if token_ids[tx] != -1:
            # Inside the if, we read from data_shared
            result_local[0] = data_shared[tx]
