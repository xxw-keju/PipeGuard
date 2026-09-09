    @T.prim_func
    def before(A: T.Tensor((64,), T.float16), C: T.Tensor((64,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for i in T.Pipelined(4, num_stages=2):
            T.copy(A[i * 16], S)
            T.copy(S, C[i * 16])
