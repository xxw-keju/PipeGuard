@T.prim_func
def vec_arith(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    Add: T.Tensor((N,), "float32"),
    Sub: T.Tensor((N,), "float32"),
    Mul: T.Tensor((N,), "float32"),
    Div: T.Tensor((N,), "float32"),
):
    for i in T.Parallel(N):
        Add[i] = A[i] + B[i]
    for i in T.Parallel(N):
        Sub[i] = A[i] - B[i]
    for i in T.Parallel(N):
        Mul[i] = A[i] * B[i]
    for i in T.Parallel(N):
        Div[i] = A[i] / B[i]
