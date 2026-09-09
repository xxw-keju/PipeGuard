    @T.prim_func
    def before(A: T.Buffer((1,), "float16")):
        for i in T.serial(1 << 20):
            T.evaluate(
                T.tvm_access_ptr(
                    T.type_annotation(T.float16),
                    A.data,
                    i * 2,
                    1,
                    1,
                )
            )
