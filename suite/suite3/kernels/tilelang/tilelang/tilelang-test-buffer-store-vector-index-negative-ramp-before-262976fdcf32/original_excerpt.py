    @T.prim_func
    def before(A: T.Tensor((1024,), T.float32)):
        vec = T.Ramp(-4, 1, 4)  # indices: [-4, -3, -2, -1]
        values = T.Ramp(0.0, 1.0, 4)  # values: [0.0, 1.0, 2.0, 3.0]
        A[vec] = values
