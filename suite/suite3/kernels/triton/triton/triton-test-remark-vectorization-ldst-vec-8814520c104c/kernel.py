# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/test_perf_warning.py#L99-L120
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def ldst_vec(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, XBLOCK: tl.constexpr):
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        x0 = xindex % 9
        x2 = (xindex // 3456) % 512
        x1 = (xindex // 9) % 384
        x4 = xindex
        tmp0 = tl.load(in_ptr0 + (x2 + (512 * x0)), None, eviction_policy="evict_last")
        tmp1 = tmp0 + 520
        tmp2 = tmp0 < 0
        tmp3 = tl.where(tmp2, tmp1, tmp0)
        tmp9 = (-4) + tmp3
        tmp12 = tl.full([1], 512, tl.int64)
        tmp14 = tmp9 < tmp12
        tmp16 = tl.load(in_ptr3 + (x1), tmp14, eviction_policy="evict_last", other=0.0)
        tmp18 = tmp16.to(tl.float32)
        tmp19 = tmp18.to(tl.float32)
        tmp20 = tl.full(tmp19.shape, 0.0, tmp19.dtype)
        tmp21 = tl.where(tmp14, tmp19, tmp20)
        tmp22 = tmp21.to(tl.float32)
        tl.store(out_ptr0 + (x4), tmp22, None)
    return ldst_vec


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
