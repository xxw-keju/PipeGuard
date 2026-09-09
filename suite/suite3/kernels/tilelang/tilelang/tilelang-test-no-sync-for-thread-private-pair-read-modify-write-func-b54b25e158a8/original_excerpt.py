    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 64)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        x1: T.float32 = s[tx * 2]
        x2: T.float32 = s[tx * 2 + 1]
        s[tx * 2] = x1 + x2
        s[tx * 2 + 1] = x1 - x2
