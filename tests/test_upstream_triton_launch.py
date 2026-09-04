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
        "kernel[True](K=1)",
        "kernel[(1,)](K=value)",
        "kernel[(1,)](K=1, num_ctas=2)",
        "kernel[(1,)](K=1, num_warps=3)",
        "kernel[(1,)](K=1, unknown_option=4)",
        "kernel[(1,)](**{'K': 1})",
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


def test_launch_rejects_runtime_arguments_even_when_call_value_is_literal():
    assert _recipes('''
import triton
import triton.language as tl

@triton.jit
def kernel(pointer, K: tl.constexpr):
    pass

kernel[(1,)](0, K=1)
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


def test_launch_rejects_autotuned_candidate_without_static_winner():
    assert _recipes("""
import triton
import triton.language as tl

@triton.autotune(configs=[triton.Config({"BLOCK": 32})], key=[])
@triton.jit
def kernel(BLOCK: tl.constexpr):
    pass

kernel[(1,)]()
""") == []
