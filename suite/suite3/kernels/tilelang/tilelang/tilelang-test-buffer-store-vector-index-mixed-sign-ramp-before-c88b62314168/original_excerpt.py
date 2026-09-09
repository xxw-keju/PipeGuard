    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-2, 1, 4)  # indices: [-2, -1, 0, 1]
        values = T.Ramp(0.0, 1.0, 4)
        A[vec] = values
