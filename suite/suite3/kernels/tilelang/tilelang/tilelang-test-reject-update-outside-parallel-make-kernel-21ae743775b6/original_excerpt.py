        @T.prim_func
        def kernel(A: T.Tensor((8,), T.float32), B: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=128):
                src = T.alloc_fragment((8,), T.float32)
                T.copy(A, src)
                acc = T.alloc_reducer((1,), T.float32, op="sum")
                T.reducer_init(acc)
                T.reducer_update(acc[0], src[0])  # thread-uniform: ambiguous
                result = T.alloc_fragment((1,), T.float32)
                T.finalize_reducer(acc, result)
                if T.get_thread_binding() == 0:
                    B[0] = result[0]
