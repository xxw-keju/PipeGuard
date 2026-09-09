    @T.prim_func(private=True)
    def func():
        temp_shared = T.alloc_buffer([128], dtype="float32", scope="shared")
        result_local = T.alloc_buffer([1], dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        result_local[0] = T.float32(0)
        # Write to shared memory
        temp_shared[tx] = T.float32(tx)
        # Outer non-uniform condition
        if tx < 64:
            # Inner condition (also non-uniform)
            if tx < 32:
                # Sync needed here must be hoisted all the way out
                result_local[0] = temp_shared[tx + 64]
