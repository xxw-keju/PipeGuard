    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            # LetStmt value has broadcasts
            val: T.float8_e4m3fnx8 = T.Broadcast(T.float8_e4m3fn(1.2), 8) + T.Broadcast(T.float8_e4m3fn(3.4), 8)
            # Body is a BufferStore with additional broadcasts
            A_shared[0:8] = val + T.Broadcast(T.float8_e4m3fn(5.6), 8)
