    @T.prim_func
    def kernel(A: T.Tensor((BC,), torch.float32), B: T.Tensor((1,), torch.float32)):
        with T.Kernel(1, threads=64) as _:
            src = T.alloc_fragment((BC,), torch.float32)
            T.copy(A, src)
            acc = T.alloc_reducer((1,), torch.float32, op="sum")
            T.reducer_init(acc)
            for i in T.Parallel(BC):
                T.reducer_update(acc[0], src[i])
            result = T.alloc_fragment((1,), torch.float32)
            T.finalize_reducer(acc, result)
            if T.get_thread_binding() == 0:
                B[0] = result[0]
