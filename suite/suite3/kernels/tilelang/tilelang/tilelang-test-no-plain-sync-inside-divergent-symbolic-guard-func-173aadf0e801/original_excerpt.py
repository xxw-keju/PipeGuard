    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((128,), dtype="float32", scope="shared")
        bx = T.launch_thread("blockIdx.x", 32)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if bx * 4 + tx // 32 < n:
            v: T.float32 = s[tx]
            s[tx] = v * T.float32(2)
