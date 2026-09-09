@libentry()
@triton.jit
def _embedding_bag_dense_backward_kernel(
    grad_ptr,
    indices_ptr,
    offset2bag_ptr,
    bag_size_ptr,
    grad_weight_ptr,
    num_weights,
    padding_idx,
    mode,
    has_per_sample_weights,
    per_sample_weights_ptr,
    scale_grad_by_freq,
    BLOCK_D: tl.constexpr,
    EMBED_DIM: tl.constexpr,
):
    """Kernel for _embedding_bag_dense_backward.

    Args:
        grad_ptr: gradient tensor of shape (num_bags, embedding_dim)
        indices_ptr: indices into embedding table, shape (num_samples,)
        offset2bag_ptr: maps sample position to bag index, shape (num_samples,)
        bag_size_ptr: number of samples in each bag, shape (num_bags,)
        grad_weight_ptr: output gradient for embedding table, shape (num_weights, embedding_dim)
        num_weights: number of rows in embedding table
        padding_idx: padding index to ignore
        mode: 0=mean, 1=sum
        has_per_sample_weights: whether per_sample_weights are provided
        per_sample_weights_ptr: optional per-sample weights
        scale_grad_by_freq: whether to scale by word frequency
        BLOCK_D: block size for embedding dimension
        EMBED_DIM: embedding dimension

    Note on scaling:
        For mode=0 (mean): grad is already scaled by bag_size, so no division needed
        For mode=1 (sum): grad needs to be divided by bag_size
    """
    pid = tl.program_id(0)
    pid_d = tl.program_id(1)

    offs_d = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
    mask_d = offs_d < EMBED_DIM

    # Load index and bag for this sample
    idx = tl.load(indices_ptr + pid).to(tl.int32)
    bag_idx = tl.load(offset2bag_ptr + pid).to(tl.int32)

    # Check validity
    valid = (idx != padding_idx) & (idx >= 0) & (idx < num_weights)

    # Load bag size for this bag
    bag_size = tl.load(bag_size_ptr + bag_idx).to(tl.float32)
    bag_size = tl.where(bag_size == 0.0, 1.0, bag_size)

    # Load gradient for this bag
    go_ptrs = grad_ptr + bag_idx * EMBED_DIM + offs_d
    go = tl.load(go_ptrs, mask=mask_d, other=0.0).to(tl.float32)

    # Compute scale factor based on mode
    # For mode=0 (mean): grad is already scaled by bag_size from autograd, so scale=1
    # For mode=1 (sum): grad needs to be divided by bag_size
    if mode == 0:
        scale = 1.0
    else:
        # Sum mode: divide by bag size
        scale = 1.0 / bag_size

    # Handle per_sample_weights if provided
    if has_per_sample_weights:
        pw = tl.load(per_sample_weights_ptr + pid).to(tl.float32)
        scale = scale * pw

    # Handle scale_grad_by_freq
    if scale_grad_by_freq:
        # Count occurrences of this index in offset2bag
        n_samples = tl.num_programs(0)
        cnt = 0
        for i in range(n_samples):
            loaded_idx = tl.load(indices_ptr + i).to(tl.int32)
            cnt = cnt + (loaded_idx == idx)
        if cnt > 1:
            scale = scale / tl.cast(cnt, tl.float32)

    # Apply scale to gradient
    go = go * scale

    # Store to output using atomic add
    gw_ptrs = grad_weight_ptr + idx * EMBED_DIM + offs_d
    mask = mask_d & valid
    tl.atomic_add(gw_ptrs, go, mask=mask)
