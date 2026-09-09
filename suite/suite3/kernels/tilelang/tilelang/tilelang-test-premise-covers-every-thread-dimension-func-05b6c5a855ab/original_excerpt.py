    @T.prim_func(private=True)
    def func(n: T.int32):
        s = T.alloc_buffer((256,), dtype="float32", scope="shared")
        l = T.alloc_buffer((1,), dtype="float32", scope="local")
        bx = T.launch_thread("blockIdx.x", 1)
        tx = T.launch_thread("threadIdx.x", 128)
        ty = T.launch_thread("threadIdx.y", 2)
        tz = T.launch_thread("threadIdx.z", 1)
        with T.Assert(n % 128 == 0, "n must be a multiple of the x extent"):
            if tx < n:
                s[ty * 128 + tx] = T.cast(tx, "float32")
                l[0] = s[ty * 128 + (tx + 64) % 128]
