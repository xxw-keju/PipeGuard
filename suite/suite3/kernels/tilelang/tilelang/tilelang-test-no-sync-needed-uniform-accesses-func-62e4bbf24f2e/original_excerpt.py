    @T.prim_func(private=True)
    def func():
        temp_local = T.alloc_buffer([1], dtype="float32", scope="local")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        temp_local[0] = T.float32(tx)
        # Non-uniform condition but no shared memory access
        if tx < 64:
            result_local[0] = temp_local[0]
