    @T.prim_func
    def after():
        with T.Kernel(8):
            A_shared = T.decl_buffer((256), T.float8_e4m3fn, scope="shared.dyn")
            # Hoisted from LetStmt value
            broadcast_var: T.float8_e4m3fn = T.float8_e4m3fn(1.2)
            broadcast_var_1: T.float8_e4m3fn = T.float8_e4m3fn(3.4)
            val: T.float8_e4m3fnx8 = T.Broadcast(broadcast_var, 8) + T.Broadcast(broadcast_var_1, 8)
            # Hoisted from BufferStore
            broadcast_var_2: T.float8_e4m3fn = T.float8_e4m3fn(5.6)
            A_shared[0:8] = val + T.Broadcast(broadcast_var_2, 8)
