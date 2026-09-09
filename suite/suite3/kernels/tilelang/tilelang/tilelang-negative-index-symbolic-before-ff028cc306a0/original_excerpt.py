@T.prim_func
def negative_index_symbolic_before(shift: T.int32, A: T.Buffer((16,), T.float32), B: T.Buffer((16,), T.float32)):
    T.func_attr({"tir.noalias": True})
    for i in T.serial(16):
        B[i] = A[shift + i]
