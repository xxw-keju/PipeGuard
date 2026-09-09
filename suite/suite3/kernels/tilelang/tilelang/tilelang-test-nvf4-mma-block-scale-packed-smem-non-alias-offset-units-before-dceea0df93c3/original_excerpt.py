    @T.prim_func
    def before(
        A: T.Tensor((16,), T.float4_e2m1fn),
        B: T.Tensor((16,), T.float4_e2m1fn),
    ):
        with T.Kernel(1, threads=32):
            a = T.alloc_shared((3,), T.float4_e2m1fn)
            b = T.alloc_shared((4,), T.float4_e2m1fn)
            a[0] = A[0]
            b[0] = A[1]
            B[0] = a[0]
            B[1] = b[0]
