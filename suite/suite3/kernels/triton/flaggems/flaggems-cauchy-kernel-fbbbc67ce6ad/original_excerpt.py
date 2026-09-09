@triton.autotune(configs=configs, key=["N"])
@triton.jit(do_not_specialize=["philox_seed", "philox_offset", "median", "sigma"])
def cauchy_kernel(
    out_ptr,
    N,
    median,
    sigma,
    philox_seed,
    philox_offset,
    BLOCK: tl.constexpr,
):
    philox_seed = philox_seed.to(tl.int64)
    philox_offset = philox_offset.to(tl.int64)
    c0 = (philox_offset & 0xFFFFFFFF).to(tl.uint32)
    c1 = ((philox_offset >> 32) & 0xFFFFFFFF).to(tl.uint32)
    i4 = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    c0 += i4
    _O = c0 * 0
    r0, r1, r2, r3 = tl.philox(philox_seed, c0, c1, _O, _O)
    r0 = uint_to_uniform_float(r0)
    r1 = uint_to_uniform_float(r1)
    r2 = uint_to_uniform_float(r2)
    r3 = uint_to_uniform_float(r3)
    c0 = uniform_to_cauchy(r0, median, sigma)
    c1 = uniform_to_cauchy(r1, median, sigma)
    c2 = uniform_to_cauchy(r2, median, sigma)
    c3 = uniform_to_cauchy(r3, median, sigma)
    off_0 = tl.program_id(0) * BLOCK * 4 + tl.arange(0, BLOCK)
    off_1 = off_0 + BLOCK
    off_2 = off_1 + BLOCK
    off_3 = off_2 + BLOCK

    tl.store(out_ptr + off_0, c0, mask=off_0 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_1, c1, mask=off_1 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_2, c2, mask=off_2 < N, eviction_policy="evict_first")
    tl.store(out_ptr + off_3, c3, mask=off_3 < N, eviction_policy="evict_first")
