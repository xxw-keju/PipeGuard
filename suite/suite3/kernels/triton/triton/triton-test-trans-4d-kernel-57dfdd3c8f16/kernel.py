# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L3696-L3714
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(In, Out,  #
               in_shape1: tl.constexpr, in_shape2: tl.constexpr, in_shape3: tl.constexpr, in_shape4: tl.constexpr,
               ou_shape1: tl.constexpr, ou_shape2: tl.constexpr, ou_shape3: tl.constexpr, ou_shape4: tl.constexpr,
               trans1: tl.constexpr, trans2: tl.constexpr, trans3: tl.constexpr, trans4: tl.constexpr):
        in_desc = tl.make_tensor_descriptor(
            base=In,
            shape=[in_shape1, in_shape2, in_shape3, in_shape4],
            strides=[in_shape4 * in_shape3 * in_shape2, in_shape4 * in_shape3, in_shape4, 1],
            block_shape=[in_shape1, in_shape2, in_shape3, in_shape4],
        )
        out_desc = tl.make_tensor_descriptor(
            base=Out,
            shape=[ou_shape1 * ou_shape2 * ou_shape3 * ou_shape4],
            strides=[1],
            block_shape=[ou_shape1 * ou_shape2 * ou_shape3 * ou_shape4],
        )
        val = in_desc.load([0, 0, 0, 0]).permute((trans1, trans2, trans3, trans4))
        out_desc.store([0], val.reshape(out_desc.block_shape))
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
