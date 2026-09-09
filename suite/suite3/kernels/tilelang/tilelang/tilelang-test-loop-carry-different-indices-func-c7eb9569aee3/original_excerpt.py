    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 1)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        for i in range(10):
            # Write to A[i], read from A[i+1]
            # After shift: comparing A[i] (write) vs A[i+2] (read from i+1 shifted)
            # No overlap, no dependency
            temp_shared[i] = T.float32(i)
            result_local[0] = result_local[0] + temp_shared[i + 1]
