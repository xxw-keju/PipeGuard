        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                for i in T.Parallel(8):
                    T.reducer_update(acc[0], src[i])
                if T.get_thread_binding() == 0:
                    B[0] = acc[0]  # reading a partial: must be rejected
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
