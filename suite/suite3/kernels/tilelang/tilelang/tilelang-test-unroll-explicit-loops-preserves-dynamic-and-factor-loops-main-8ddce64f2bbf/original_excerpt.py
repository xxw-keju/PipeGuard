        @T.prim_func
        def main(n: T.int32, A: T.Buffer((16,), "int32")):
            for i in T.unroll(2, annotations={"pragma_unroll_explicit": True}):
                A[i] = i
            for j in T.unroll(n):
                A[j] = j
            for k in T.unroll(8, annotations={"pragma_unroll_factor": 4}):
                A[k + 8] = k
