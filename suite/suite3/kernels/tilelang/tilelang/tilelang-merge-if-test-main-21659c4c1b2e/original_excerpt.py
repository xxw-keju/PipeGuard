    @T.prim_func
    def main():
        A = T.alloc_fragment((1,), T.float16)
        B = T.alloc_fragment((1,), T.float16)
        C = T.alloc_fragment((1,), T.float16)
        D = T.alloc_fragment((1,), T.float16)
        if A[0] == 0:
            A[0] = 0
        if B[0] == 0:
            B[0] = 0
        if C[0] == 0:
            C[0] = 0
        if D[0] == 0:
            D[0] = 0
