from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from upstream_extract import AliasTable, CandidateVisitor, infer_triton_recipes  # noqa: E402
from upstream_pytest_params import pytest_parameter_environments  # noqa: E402
from upstream_static_eval import literal_value  # noqa: E402


def _discover(source: str):
    tree = ast.parse(source)
    aliases = AliasTable(tree)
    visitor = CandidateVisitor("triton", "case.py", source, aliases)
    visitor.visit(tree)
    return tree, aliases, visitor.candidates


def test_literal_parametrize_emits_only_non_raises_compile_recipes():
    source = """\
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def topk_kernel(K: tl.constexpr):
    x = tl.arange(0, 8)
    tl.topk(x, K)

@pytest.mark.parametrize("k,error", [(0, "zero"), (-1, "negative"), (3, "power"), (16, "large")])
def test_invalid(k, error):
    src = ASTSource(fn=topk_kernel, signature={"K": "constexpr"}, constexprs={"K": k})
    with pytest.raises(triton.CompilationError):
        triton.compile(src, target=GPUTarget("cuda", 90, 32))

@pytest.mark.parametrize("k", [1, 8])
def test_valid(k):
    src = ASTSource(fn=topk_kernel, signature={"K": "constexpr"}, constexprs={"K": k})
    triton.compile(src, target=GPUTarget("cuda", 90, 32))
"""
    tree, aliases, candidates = _discover(source)
    recipes = infer_triton_recipes(candidates, tree, aliases, "case.py")
    assert list(recipes) == [candidates[0].candidate_id]
    assert sorted(
        recipe["compiler"]["constexprs"]["K"]
        for recipe in recipes[candidates[0].candidate_id]
    ) == [1, 8]
    assert all(
        recipe["origin"] == "static-inference:case.py:21-21"
        for recipe in recipes[candidates[0].candidate_id]
    )


def test_parametrize_rejects_indirect_dynamic_and_xfail():
    cases = [
        "@pytest.mark.parametrize('k', [1], indirect=False)\ndef test_case(k): pass\n",
        "@pytest.mark.parametrize('k', VALUES)\ndef test_case(k): pass\n",
        "@pytest.mark.xfail\n@pytest.mark.parametrize('k', [1])\ndef test_case(k): pass\n",
        "@pytest.mark.parametrize('k', [pytest.param(1, marks=pytest.mark.xfail)])\ndef test_case(k): pass\n",
        "@pytest.mark.parametrize('k', [1])\ndef test_case(k, monkeypatch): pass\n",
    ]
    for body in cases:
        tree = ast.parse("import pytest\n" + body)
        aliases = AliasTable(tree)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
        assert pytest_parameter_environments(function, aliases) == []


def test_static_literal_evaluator_rejects_executable_ast():
    for expression in ("factory()", "left + right", "module.value", "items[0]"):
        with pytest.raises(ValueError):
            literal_value(ast.parse(expression, mode="eval").body, {})


def test_tracked_topk_recipe_ids_and_values():
    recipe_root = ROOT / "corpus" / "upstream" / "triton-61c02bcd" / "recipes" / "20bc89c8423f81e4"
    expected = {
        "20bc89c8423f81e4-sm90-L42-bfe52c49.json": {"K": 1},
        "20bc89c8423f81e4-sm90-L42-661f9844.json": {"K": 8},
    }
    concrete = {path.name: json.loads(path.read_text()) for path in recipe_root.glob("*.json") if path.name != "unresolved.json"}
    assert set(concrete) == set(expected)
    assert {name: recipe["compiler"]["constexprs"] for name, recipe in concrete.items()} == expected
