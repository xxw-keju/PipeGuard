        @T.prim_func
        def subroutine(A_data: T.handle("float32")):
            T.func_attr({"global_symbol": "subroutine", "target": T.target("llvm", host="llvm")})
            T.evaluate(A_data)
