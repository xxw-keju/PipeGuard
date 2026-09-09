@T.prim_func
def negative_index_loop_before(A: T.Buffer((16,), T.float32), B: T.Buffer((4,), T.float32)):
    T.func_attr({"tir.noalias": True})
    for i in T.serial(4):
        B[i] = A[-i - 1]
