@T.prim_func
def vec_add_cuda(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    with T.Kernel(1, threads=N) as bx:
        for i in T.Parallel(N):
            B[bx * N + i] = A[bx * N + i] + 1.0  # line_marker_store_cuda
