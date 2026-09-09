# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/test_stages_inspection.py#L45-L47
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def k2():
        return
    return k2


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
