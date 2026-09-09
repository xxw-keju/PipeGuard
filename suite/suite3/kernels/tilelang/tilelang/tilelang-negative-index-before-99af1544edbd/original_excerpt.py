@T.prim_func
def negative_index_before(A: T.Buffer((16,), T.float32), B: T.Buffer((16,), T.float32)):
    T.func_attr({"tir.noalias": True})
    B[0] = A[T.int32(-1)]
