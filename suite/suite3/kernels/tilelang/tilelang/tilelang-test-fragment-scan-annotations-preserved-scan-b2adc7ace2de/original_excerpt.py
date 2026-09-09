    @T.prim_func
    def scan(A: T.Tensor((32,), T.float32)):
        with T.Kernel(1, threads=32):
            sum_frag = T.alloc_fragment((32,), T.float32)
            max_frag = T.alloc_fragment((32,), T.float32)
            T.copy(A, sum_frag)
            T.copy(A, max_frag)
            T.cumsum(sum_frag, annotations={"test.scan": "cumsum"})
            T.cummax(max_frag, annotations={"test.scan": "cummax"})
