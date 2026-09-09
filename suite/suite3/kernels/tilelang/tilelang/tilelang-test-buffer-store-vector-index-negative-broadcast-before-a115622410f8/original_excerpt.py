    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Broadcast(-1, 4)
        values = T.Broadcast(42.0, 4)
        A[vec] = values
