    @T.prim_func
    def before():
        with T.Kernel(1):
            smem = T.decl_buffer((1,), T.float16, scope="shared")
            smem[0] = T.float16(0)
            T.evaluate(T.call_extern("handle", "custom_op"))
