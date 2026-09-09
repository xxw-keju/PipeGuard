# Normalized device code for PipeGuard Suite 3.
# Source: https://github.com/triton-lang/triton/blob/61c02bcdba0ccc1cdfc89ba608aa1a1221a59e1f/python/test/unit/language/test_core.py#L5595-L5623
# The exact upstream excerpt is stored in original_excerpt.py.

import triton

import triton.language as tl

def _resolve_kernel(**pipeguard_bindings):
    @triton.jit
    def kernel(Cond, XTrue, XFalse, Ret, IfType: tl.constexpr, BoolVar: tl.constexpr, StaticValue: tl.constexpr):
        pid = tl.program_id(0)
        cond = tl.load(Cond)
        if IfType == "if":
            if pid % 2 == 0:  # eq
                tl.store(Ret, tl.load(XTrue))
            elif 1 == pid % 2:  # req
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_dynamic":
            val = tl.load(XTrue) if pid % 2 == 0 else tl.load(XFalse)
            tl.store(Ret, val)
        elif IfType == "if_exp_dynamic_constexpr":
            val = 3.14 if pid % 2 == 0 else tl.load(XFalse)
            tl.store(Ret, val)
        elif IfType == "if_exp_dynamic_void":
            tl.store(Ret, tl.load(XTrue)) if pid % 2 == 0 else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_exp_static":
            tl.store(Ret, tl.load(XTrue)) if BoolVar else tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_dynamic":
            if BoolVar and (1 != pid % 2 and pid % 2 != 1):  # rne and ne
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
        elif IfType == "if_and_static":
            if StaticValue != 0 and StaticValue != 0:
                tl.store(Ret, tl.load(XTrue))
            else:
                tl.store(Ret, tl.load(XFalse))
    return kernel


def build_kernel(case):
    """Construct the DSL-native device entry without compiling or launching it."""
    arguments = dict(case.get("arguments", {}))
    build = case.get("build", {})
    arguments.update(build.get("resolver_kwargs", {}))
    return _resolve_kernel(**arguments)
