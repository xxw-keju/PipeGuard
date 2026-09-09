    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            A_shared[0:8] = T.Broadcast(T.float8_e4m3fn(1.2), 8) + T.Broadcast(T.float8_e4m3fn(3.4), 8)
