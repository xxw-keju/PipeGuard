@libentry()
@triton.jit
def cholesky_kernel(A, L, N, batch_stride, stride_a, stride_l):
    """Cholesky decomposition kernel.

    Each program computes one matrix in the batch.
    """
    pid = program_id(0)

    # Each matrix is N*N elements; batch_stride is the stride between matrices
    a_offset = pid * batch_stride
    l_offset = pid * batch_stride

    # Sequential Cholesky: L[i,j] for j <= i
    for i in range(N):
        for j in range(i + 1):
            # Accumulate in the input dtype to support float32 and float64
            sum_val = tl.zeros((), dtype=A.dtype.element_ty)

            # sum(L[i,k] * L[j,k]) for k in [0, j)
            if j > 0:
                for k in range(j):
                    sum_val = sum_val + tl.load(
                        L + l_offset + i * stride_l + k
                    ) * tl.load(L + l_offset + j * stride_l + k)

            if j == i:
                # Diagonal: L[i,i] = sqrt(A[i,i] - sum)
                a_diag = tl.load(A + a_offset + i * stride_a + i)
                result = tl.sqrt(a_diag - sum_val)
                tl.store(L + l_offset + i * stride_l + i, result)
            else:
                # Off-diagonal: L[i,j] = (A[i,j] - sum) / L[j,j]
                a_val = tl.load(A + a_offset + i * stride_a + j)
                l_diag = tl.load(L + l_offset + j * stride_l + j)
                result = (a_val - sum_val) / l_diag
                tl.store(L + l_offset + i * stride_l + j, result)
