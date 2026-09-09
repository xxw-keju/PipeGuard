    @T.prim_func
    def main(v_thread: T.int32, out: T.Tensor((1,), "int32")):
        with T.Kernel(1):
            state = T.alloc_fragment((1,), "int32")
            state[0] = v_thread
            out[0] = state[0]
