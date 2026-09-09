        @T.prim_func
        def main(A: T.Buffer((16,), "float32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A_flat = T.decl_buffer((16,), "float32", data=A.data)
                A_flat[0] = T.float32(i)
