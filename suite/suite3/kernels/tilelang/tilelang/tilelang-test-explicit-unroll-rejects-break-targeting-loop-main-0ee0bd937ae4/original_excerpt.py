        @T.prim_func
        def main(A: T.Buffer((2,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
                if A[i] != 0:
                    T.evaluate(T.call_intrin("handle", tvm.tirx.op.Op.get("tl.loop_break")))
