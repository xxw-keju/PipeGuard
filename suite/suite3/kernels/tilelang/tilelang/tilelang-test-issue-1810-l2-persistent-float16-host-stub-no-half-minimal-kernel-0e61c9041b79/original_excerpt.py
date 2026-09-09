    @T.prim_func
    def minimal_kernel(A: T.Buffer((1,), "float16")):
        with T.Kernel():
            T.annotate_l2_hit_ratio({A: 0.9})
            T.evaluate(0)
