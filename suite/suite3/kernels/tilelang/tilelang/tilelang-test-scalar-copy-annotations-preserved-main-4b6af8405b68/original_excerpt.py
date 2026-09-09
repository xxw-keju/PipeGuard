    @T.prim_func
    def main(A: T.Tensor((1,), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(1, threads=32):
            A_shared = T.alloc_shared((1,), T.float32)
            T.copy(A[0], B[0], annotations={"test.copy": "sync"})
            T.async_copy(A[0], A_shared[0], annotations={"test.copy": "async"})
