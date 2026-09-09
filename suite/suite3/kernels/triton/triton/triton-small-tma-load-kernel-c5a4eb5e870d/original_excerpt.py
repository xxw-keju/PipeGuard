@triton.jit
def small_tma_load_kernel(input_desc, output_ptr, N: tl.constexpr):
    offsets = tl.arange(0, 4)
    for block_start in tl.range(0, N, 4, num_stages=3):
        value = input_desc.load([block_start])
        tl.store(output_ptr + block_start + offsets, value)
