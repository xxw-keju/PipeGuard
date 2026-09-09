@libentry()
@triton.jit
def triangular_inverse_kernel(
    L_ptr,
    L_inv_ptr,
    N,
    batch_stride,
    stride_row,
    stride_col,
    upper: tl.constexpr,
):
    """Compute the inverse of a triangular matrix in-place per batch element.

    Each program handles one matrix in the batch.
    For lower triangular: forward substitution column by column.
    For upper triangular: back substitution column by column.
    """
    pid = program_id(0)
    base = pid * batch_stride

    if upper:
        # Back substitution for upper triangular U
        # Process columns from right to left
        for j_rev in range(N):
            j = N - 1 - j_rev
            # Diagonal: U_inv[j,j] = 1 / U[j,j]
            u_jj = tl.load(L_ptr + base + j * stride_row + j * stride_col)
            inv_jj = 1.0 / u_jj
            tl.store(L_inv_ptr + base + j * stride_row + j * stride_col, inv_jj)

            # Rows above diagonal: i < j
            for i_off in range(N):
                i = j - 1 - i_off
                if i >= 0:
                    # U_inv[i,j] = -sum(U[i,k]*U_inv[k,j], k=i+1..j) / U[i,i]
                    acc = tl.zeros((), dtype=L_ptr.dtype.element_ty)
                    for k in range(N):
                        if k > i and k <= j:
                            u_ik = tl.load(
                                L_ptr + base + i * stride_row + k * stride_col
                            )
                            inv_kj = tl.load(
                                L_inv_ptr + base + k * stride_row + j * stride_col
                            )
                            acc += u_ik * inv_kj
                    u_ii = tl.load(L_ptr + base + i * stride_row + i * stride_col)
                    result = -acc / u_ii
                    tl.store(
                        L_inv_ptr + base + i * stride_row + j * stride_col,
                        result,
                    )
    else:
        # Forward substitution for lower triangular L
        # Process columns from left to right
        for j in range(N):
            # Diagonal: L_inv[j,j] = 1 / L[j,j]
            l_jj = tl.load(L_ptr + base + j * stride_row + j * stride_col)
            inv_jj = 1.0 / l_jj
            tl.store(L_inv_ptr + base + j * stride_row + j * stride_col, inv_jj)

            # Rows below diagonal: i > j
            for i in range(N):
                if i > j:
                    # L_inv[i,j] = -sum(L[i,k]*L_inv[k,j], k=j..i-1) / L[i,i]
                    acc = tl.zeros((), dtype=L_ptr.dtype.element_ty)
                    for k in range(N):
                        if k >= j and k < i:
                            l_ik = tl.load(
                                L_ptr + base + i * stride_row + k * stride_col
                            )
                            inv_kj = tl.load(
                                L_inv_ptr + base + k * stride_row + j * stride_col
                            )
                            acc += l_ik * inv_kj
                    l_ii = tl.load(L_ptr + base + i * stride_row + i * stride_col)
                    result = -acc / l_ii
                    tl.store(
                        L_inv_ptr + base + i * stride_row + j * stride_col,
                        result,
                    )
