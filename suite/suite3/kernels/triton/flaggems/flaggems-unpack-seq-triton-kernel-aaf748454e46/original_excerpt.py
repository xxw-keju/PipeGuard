@triton.jit
def _unpack_seq_triton_kernel(
    packed_ptr,  # [B, Lmax, D]
    out_ptr,  # [N, D]
    lengths_ptr,  # *i32, [B]
    B: tl.constexpr,
    Lmax: tl.constexpr,
    D: tl.constexpr,
    BLOCK_B: tl.constexpr,
    BLOCK_T: tl.constexpr,  # timesteps per program
    BLOCK_D: tl.constexpr,  # features per program
):
    pid_b = tl.program_id(0)  # batch id
    pid_t = tl.program_id(1)  # block over time dimension
    pid_d = tl.program_id(2)  # block over feature dimension
    off_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)  # [BLOCK_T]
    off_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)  # [BLOCK_D]

    # bounds: compute start from cumulative lengths
    off_b = tl.arange(0, BLOCK_B)
    prev_lengths = tl.load(lengths_ptr + off_b, mask=off_b < pid_b, other=0)
    in_start = tl.sum(prev_lengths, axis=0)
    seq_len = tl.load(lengths_ptr + pid_b)

    # valid time positions for this block
    t_mask = off_t < Lmax
    valid_row = (off_t < seq_len) & t_mask

    # compute output row indices for valid (b, t)
    out_row = in_start + off_t

    # Pointers
    # packed_ptr: row-major [B, Lmax, D]
    packed_row_ptr = packed_ptr + (pid_b * Lmax + off_t)[:, None] * D + off_d[None, :]

    # out_ptr: row-major [N, D]
    out_row_ptr = out_ptr + out_row[:, None] * D + off_d[None, :]

    # Load from packed tensor and store to output
    d_mask = off_d[None, :] < D
    packed_vals = tl.load(packed_row_ptr, mask=valid_row[:, None] & d_mask)
    tl.store(out_row_ptr, packed_vals, mask=valid_row[:, None] & d_mask)
