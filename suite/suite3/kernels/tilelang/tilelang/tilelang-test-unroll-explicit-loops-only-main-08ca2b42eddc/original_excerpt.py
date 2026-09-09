        @T.prim_func
        def main(A: T.Buffer((4,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(2, annotations={"pragma_unroll_explicit": False}):
                A[j + 2] = j
