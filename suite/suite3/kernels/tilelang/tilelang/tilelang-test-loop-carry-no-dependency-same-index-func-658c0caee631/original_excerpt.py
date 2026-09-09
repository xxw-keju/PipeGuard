    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for i in range(10):
            # Each iteration writes to A[tx], then reads from A[tx]
            # No loop-carried dependency because different iterations
            # access different locations
            temp_shared[tx] = T.float32(i)
            result_local[0] = result_local[0] + temp_shared[tx]
