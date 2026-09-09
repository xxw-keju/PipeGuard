    @T.prim_func
    def before():
        with T.Kernel(8):
            A_global = T.decl_buffer((128,), T.float16, scope="global")
            T.evaluate(T.call_intrin("handle", tirx.op.Op.get("tl.tma_store"), A_global.data))
