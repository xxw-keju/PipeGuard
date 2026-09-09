    @T.prim_func
    def before():
        with T.Kernel(8):
            A_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            B_shared = T.decl_buffer((1024,), T.uint8, scope="shared.dyn")
            T.ptx_cp_async(
                T.tvm_access_ptr(T.type_annotation(T.uint8), A_shared.data, 0, 16, 2),
                T.tvm_access_ptr(T.type_annotation(T.uint8), B_shared.data, 0, 16, 1),
                16,
            )
            T.fence_proxy_async()
            T.call_extern("handle", "generic_op")
