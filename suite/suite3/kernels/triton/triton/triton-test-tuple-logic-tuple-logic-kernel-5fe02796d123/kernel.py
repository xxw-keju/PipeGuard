# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_tuple.py#L353-L367
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def tuple_logic_kernel():

        # arity-2 BoolOps:
        tl.static_assert(((3, 4) or (5, 6)) == (3, 4))
        tl.static_assert(((3, 4) and (5, 6)) == (5, 6))
        tl.static_assert(((3, 4) and ()) == ())
        tl.static_assert((() or (5, 6)) == (5, 6))

        # arity-3 BoolOps:
        tl.static_assert(((1, 2) and (3, 4) and (5, 6)) == (5, 6))
        tl.static_assert(((1, 2) or (3, 4) or (5, 6)) == (1, 2))

        # constexpr short-circuiting over dynamic argument:
        tl.static_assert((() and tl.program_id(0)) == ())
    return tuple_logic_kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
