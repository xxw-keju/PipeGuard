@triton.jit
def stack_copy_func_kernel_4(
    out_ptr,
    in_ptr_a,
    in_ptr_b,
    in_ptr_c,
    in_ptr_d,
    dim_size_out,
    dim_prod_post,
    dim_offset_a,
    dim_offset_b,
    dim_offset_c,
    dim_offset_d,
    total_elements_a,
    total_elements_b,
    total_elements_c,
    total_elements_d,
    BLOCK_X: tl.constexpr,
):
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)

    if pid_y == 0:
        in_ptr = in_ptr_a
        dim_offset = dim_offset_a
        total_elements = total_elements_a
    elif pid_y == 1:
        in_ptr = in_ptr_b
        dim_offset = dim_offset_b
        total_elements = total_elements_b
    elif pid_y == 2:
        in_ptr = in_ptr_c
        dim_offset = dim_offset_c
        total_elements = total_elements_c
    else:
        in_ptr = in_ptr_d
        dim_offset = dim_offset_d
        total_elements = total_elements_d

    block_start = pid_x.to(tl.int64) * BLOCK_X
    offsets = tl.arange(0, BLOCK_X).to(tl.int64)
    idx = block_start + offsets
    scalar_zero = offsets * 0

    dim_size_out = dim_size_out + scalar_zero
    dim_prod_post = dim_prod_post + scalar_zero
    dim_offset = dim_offset + scalar_zero
    total_elements = total_elements + scalar_zero

    mask = idx < total_elements

    pre_idx = idx // dim_prod_post
    post_idx = idx % dim_prod_post

    out_idx = (
        pre_idx * dim_size_out * dim_prod_post + dim_offset * dim_prod_post + post_idx
    )

    data = tl.load(in_ptr + idx, mask=mask)
    tl.store(out_ptr + out_idx, data, mask=mask)
