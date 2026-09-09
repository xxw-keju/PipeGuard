    @triton.jit
    def kernel(output_ptr, n_elements, BLOCK_SIZE: tl.constexpr, CONSTANT_FIELD: tl.constexpr):
        offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        if CONSTANT_FIELD == "value":
            value = 1
            output = tl.full([BLOCK_SIZE], value=value, dtype=value.dtype)
            mask = offsets < n_elements
        elif CONSTANT_FIELD == "mask":
            output = offsets < n_elements
            mask = False
        tl.store(output_ptr + offsets, output, mask=mask)
