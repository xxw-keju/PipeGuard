# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/tutorials/11-programmatic-dependent-launch.py#L27-L55
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

@triton.jit
def add_kernel(x_ptr,  #
               y_ptr,  #
               output_ptr,  #
               n_elements,  #
               BLOCK_SIZE: tl.constexpr,  #
               USE_GDC: tl.constexpr,  #
               ):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    if USE_GDC:
        # GDC wait waits for ALL programs in the the prior kernel to complete before continuing.
        # This ensures any memory operations happen before the wait in program order,
        # e.g. if the prior kernel writes to x or y the new values will be visible.
        tl.extra.cuda.gdc_wait()

    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    if USE_GDC:
        # GDC launch dependents hints the runtime system to launch dependent kernels.
        # These dependent kernels must also be launched with PDL enabled.
        # Once GDC launch has been issued by ALL programs or
        # programs have finished, the dependent grid can begin if there are enough resources.
        # Note: this by itself provides no additional memory-ordering guarentees, unlike `gdc_wait`
        tl.extra.cuda.gdc_launch_dependents()
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)


def _resolve_kernel(**pipeguard_bindings):
    return add_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
