    @T.prim_func(private=True)
    def func():
        token_ids = T.alloc_buffer([128], dtype="int32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for k in range(2):
            # Write to shared memory
            token_ids[tx] = T.int32(k - 2)
            # Non-uniform if inside loop
            if token_ids[tx] >= 0:
                result_local[0] = T.float32(1)
