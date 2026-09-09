    @T.prim_func(private=True)
    def func():
        s = T.alloc_buffer((64,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 48)
        ty = T.launch_thread("threadIdx.y", 1)
        tz = T.launch_thread("threadIdx.z", 1)
        if tx >= 32:
            l[0] = T.float32(0)
        else:
            s[tx] = T.cast(tx, "float32")
            l[0] = s[31 - tx]
