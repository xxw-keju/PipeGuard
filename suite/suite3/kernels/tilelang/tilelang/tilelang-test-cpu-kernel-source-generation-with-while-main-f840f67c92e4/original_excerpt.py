    @T.prim_func
    def main(flag: T.Buffer((1,), "int32"), out: T.Buffer((1,), "int32")):
        with T.Kernel(1):
            state = T.alloc_fragment((1,), "int32")
            state[0] = 0

            with T.While(state[0] == 0):
                state[0] = flag[0]

            out[0] = state[0]
