from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from upstream_extract import AliasTable, CandidateVisitor, infer_triton_recipes  # noqa: E402


def _recipes(source: str) -> list[dict]:
    tree = ast.parse(source)
    aliases = AliasTable(tree)
    visitor = CandidateVisitor("triton", "launch.py", source, aliases)
    visitor.visit(tree)
    inferred = infer_triton_recipes(visitor.candidates, tree, aliases, "launch.py")
    return [recipe for recipes in inferred.values() for recipe in recipes]


def test_launch_infers_all_constexpr_bracket_and_warmup_recipes():
    recipes = _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(K: tl.constexpr = 4):
    pass

alias = kernel
alias[(1, 1)](K=8, num_warps=4, num_stages=2, maxnreg=64)
kernel.warmup(grid=[1], K=4)
''')
    assert len(recipes) == 2
    assert {recipe["compiler"]["constexprs"]["K"] for recipe in recipes} == {4, 8}
    bracket = next(recipe for recipe in recipes if recipe["compiler"]["constexprs"]["K"] == 8)
    assert bracket["compiler"] == {
        "signature": {"K": "constexpr"},
        "constexprs": {"K": 8},
        "attrs": {},
        "options": {"num_warps": 4, "num_stages": 2, "maxnreg": 64},
        "grid_ctas": 1,
        "binding_evidence": None,
    }
    assert all(
        recipe["target"] == {"backend": "cuda", "arch": "sm90", "warp_size": 32}
        for recipe in recipes
    )
    assert all(recipe["origin"].startswith("profile-sm90-launch-inference:launch.py:") for recipe in recipes)


@pytest.mark.parametrize(
    "call",
    [
        "kernel[grid](K=1)",
        "kernel[(2,)](K=1)",
        "kernel[(1,)](K=value)",
        "kernel[(1,)](K=1, num_ctas=2)",
        "kernel[(1,)](K=1, num_warps=3)",
        "kernel[(1,)](K=1, unknown_option=4)",
        "kernel[*dims](K=1)",
        "kernel.warmup(K=1)",
        "kernel.warmup(K=1, grid=(1,), num_ctas=2)",
    ],
)
def test_launch_rejects_unclosed_or_non_single_cta_calls(call: str):
    source = f'''
import triton
import triton.language as tl

@triton.jit
def kernel(K: tl.constexpr):
    pass

{call}
'''
    assert _recipes(source) == []


def test_launch_infers_native_scalar_specialization_and_literal_splat():
    recipes = _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(value, K: tl.constexpr):
    pass

kernel[(1,)](0, **{"K": 1})
''')
    assert len(recipes) == 1
    assert recipes[0]["compiler"] == {
        "signature": {"value": "i32", "K": "constexpr"},
        "constexprs": {"K": 1},
        "attrs": {"value": [["tt.divisibility", 16]]},
        "options": {},
        "grid_ctas": 1,
        "binding_evidence": None,
    }


def test_launch_rejects_boolean_grid():
    assert _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(K: tl.constexpr):
    pass

kernel[True](K=1)
''') == []


def test_launch_rejects_varargs_and_missing_constexprs():
    assert _recipes('''
import triton
import triton.language as tl

@triton.jit
def vararg_kernel(K: tl.constexpr, *values):
    pass

@triton.jit
def missing_kernel(K: tl.constexpr, *, BLOCK: tl.constexpr):
    pass

vararg_kernel[(1,)](1)
missing_kernel[(1,)](1)
''') == []


def test_launch_does_not_extract_expected_failure_path():
    assert _recipes('''
import pytest
import triton
import triton.language as tl

@triton.jit
def kernel(K: tl.constexpr):
    pass

def test_invalid():
    with pytest.raises(Exception):
        kernel[(1,)](K=1)
''') == []


def test_launch_expands_all_literal_autotune_configs_without_claiming_winner():
    recipes = _recipes("""
import triton
import triton.language as tl

@triton.autotune(configs=[
    triton.Config({"BLOCK": 32}, num_warps=4),
    triton.Config({"BLOCK": 64}, num_warps=8),
], key=[])
@triton.jit
def kernel(BLOCK: tl.constexpr):
    pass

kernel[(1,)]()
""")
    assert len(recipes) == 2
    assert {recipe["compiler"]["constexprs"]["BLOCK"] for recipe in recipes} == {32, 64}
    assert {recipe["compiler"]["options"]["num_warps"] for recipe in recipes} == {4, 8}
    assert all(
        recipe["origin"].startswith("profile-sm90-autotune-expansion:")
        for recipe in recipes
    )
    assert all("winner" not in recipe["origin"] for recipe in recipes)


def test_launch_rejects_autotune_pruning_and_config_hooks():
    assert _recipes("""
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK": 32})],
    key=[],
    prune_configs_by={"early_config_prune": prune},
)
@triton.jit
def kernel(BLOCK: tl.constexpr):
    pass

kernel[(1,)]()
""") == []
    assert _recipes("""
import triton
import triton.language as tl

@triton.autotune(
    configs=[triton.Config({"BLOCK": 32}, pre_hook=hook)],
    key=[],
)
@triton.jit
def kernel(BLOCK: tl.constexpr):
    pass

kernel[(1,)]()
""") == []


def test_launch_evaluates_closed_callable_grid_with_static_metadata():
    recipes = _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(size, BLOCK: tl.constexpr):
    pass

grid = lambda meta: (triton.cdiv(meta["size"], meta["BLOCK"]),)
kernel[grid](1, BLOCK=32)
''')
    assert len(recipes) == 1
    assert recipes[0]["compiler"]["signature"] == {
        "size": "constexpr",
        "BLOCK": "constexpr",
    }
    assert recipes[0]["compiler"]["constexprs"] == {"size": 1, "BLOCK": 32}


def test_launch_evaluates_closed_grid_function_and_rejects_multi_cta_result():
    accepted = _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(size, BLOCK: tl.constexpr):
    pass

def grid(meta):
    return (triton.cdiv(meta["size"], meta["BLOCK"]), 1)

kernel[grid](16, BLOCK=32)
''')
    assert len(accepted) == 1
    rejected = _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(size, BLOCK: tl.constexpr):
    pass

grid = lambda meta: (triton.cdiv(meta["size"], meta["BLOCK"]),)
kernel[grid](33, BLOCK=32)
''')
    assert rejected == []


def test_launch_rejects_callable_grid_with_unknown_or_executable_effects():
    assert _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(BLOCK: tl.constexpr):
    pass

grid = lambda meta: factory(meta["BLOCK"])
kernel[grid](BLOCK=32)
''') == []


def test_launch_evaluates_pure_heuristics_in_wrapper_order():
    recipes = _recipes('''
import triton
import triton.language as tl

@triton.heuristics({"BLOCK": lambda args: triton.next_power_of_2(args["N"])})
@triton.heuristics({"EVEN": lambda args: args["BLOCK"] % 2 == 0})
@triton.jit
def kernel(N, BLOCK: tl.constexpr, EVEN: tl.constexpr):
    pass

kernel[(1,)](17, BLOCK=999)
''')
    assert len(recipes) == 1
    compiler = recipes[0]["compiler"]
    assert compiler["signature"] == {
        "N": "i32",
        "BLOCK": "constexpr",
        "EVEN": "constexpr",
    }
    assert compiler["constexprs"] == {"BLOCK": 32, "EVEN": True}
    assert compiler["attrs"] == {"N": []}


def test_launch_rejects_heuristic_that_reads_unknown_runtime_object():
    assert _recipes('''
import triton
import triton.language as tl

@triton.heuristics({"EVEN": lambda args: args["pointer"].data_ptr() % 2 == 0})
@triton.jit
def kernel(pointer, EVEN: tl.constexpr):
    pass

kernel[(1,)](tensor)
''') == []


def test_launch_rejects_unknown_autotune_fields_and_duplicate_heuristics():
    assert _recipes("""
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({"BLOCK": 32})], key=[], unknown=True)
@triton.jit
def kernel(BLOCK: tl.constexpr):
    pass

kernel[(1,)]()
""") == []
    assert _recipes("""
import triton
import triton.language as tl

@triton.heuristics({
    "BLOCK": lambda args: 32,
    "BLOCK": lambda args: 64,
})
@triton.jit
def kernel(BLOCK: tl.constexpr):
    pass

kernel[(1,)]()
""") == []
