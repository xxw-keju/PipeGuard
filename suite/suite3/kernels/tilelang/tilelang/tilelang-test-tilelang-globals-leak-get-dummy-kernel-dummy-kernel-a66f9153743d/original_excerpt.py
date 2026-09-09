        @T.prim_func
        def dummy_kernel(
            a: T.Tensor[(1,), T.float32],
        ):
            with T.Kernel(1) as _:
                # `ub = T.alloc_shared(...)` is an assignment that goes through
                # `__tb.bind` -> `get_parent_locals`. Without it the eager builder
                # never calls `get_parent_locals` (subscript stores and `with ... as`
                # targets do not), so this test cannot detect the frame leak.
                ub = T.alloc_shared((1,), "float32")
                T.copy(a, ub)
                T.copy(ub, a)
