    @T.prim_func
    def before(
        counter: T.Tensor((1,), T.int32),
        out: T.Tensor((8,), T.int32),
    ):
        for i in T.Pipelined(
            4,
            order=[0, 1, 2, 3],
            stage=[0, 0, 1, 1],
        ):
            snapshot: T.int32 = counter[0]
            pos: T.int32 = T.atomic_add(counter[0], 1, return_prev=True)
            out[i * 2] = snapshot
            out[i * 2 + 1] = pos
