        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                for j in T.serial(2):
                    A[i * 2 + j] = j
                    if A[i * 2 + j] != 0:
                        T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))
