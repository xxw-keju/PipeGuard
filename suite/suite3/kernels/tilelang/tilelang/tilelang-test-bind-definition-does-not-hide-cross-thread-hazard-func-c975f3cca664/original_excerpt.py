    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        idx: T.int32 = tx + 7
        s[tx] = T.cast(tx, "float32")
        l[0] = s[idx % 128]
