    @T.prim_func
    def before(A_ptr: T.handle, B_ptr: T.handle):
        A = T.match_buffer(A_ptr, (256,), dtype=T.int32)
        B = T.match_buffer(B_ptr, (256,), dtype=T.int32)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for tx in T.thread_binding(256, thread="threadIdx.x"):
                for i in T.unroll(0, 2):
                    B[tx] = A[tx]
                    if tx == 0:
                        B[i] = T.int32(1)
