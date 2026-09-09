@T.prim_func
def vec_add(A: T.Tensor((N,), "float32"), B: T.Tensor((N,), "float32")):
    for i in T.Parallel(N):
        B[i] = A[i] + 1.0  # line_marker_store
