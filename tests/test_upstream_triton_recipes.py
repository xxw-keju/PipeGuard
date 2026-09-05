from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from upstream_extract import AliasTable, CandidateVisitor, candidate_roles, infer_triton_recipes  # noqa: E402
from upstream_triton_recipes import analyze_triton_recipes, promote_inferred_root_roles  # noqa: E402
from upstream_pytest_params import pytest_parameter_environments  # noqa: E402
from upstream_static_eval import json_literal_value, literal_value  # noqa: E402


def _discover(source: str):
    tree = ast.parse(source)
    aliases = AliasTable(tree)
    visitor = CandidateVisitor("triton", "case.py", source, aliases)
    visitor.visit(tree)
    return tree, aliases, visitor.candidates


def test_inferred_astsource_recipe_promotes_indirect_compile_candidate_to_root():
    source = """\
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel(K: tl.constexpr):
    pass

source = ASTSource(fn=kernel, signature={"K": "constexpr"}, constexprs={"K": 1})
triton.compile(source, target=GPUTarget("cuda", 90, 32))
"""
    tree, aliases, candidates = _discover(source)
    roles, _dependencies = candidate_roles(candidates, tree)
    assert roles[candidates[0].candidate_id] == "UNCLASSIFIED"
    inferred = infer_triton_recipes(candidates, tree, aliases, "case.py")
    promoted = promote_inferred_root_roles(roles, inferred)
    assert promoted[candidates[0].candidate_id] == "ROOT"


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


def test_parametrize_allows_unused_fixture_but_does_not_bind_it():
    tree = ast.parse(
        "import pytest\n"
        "@pytest.mark.parametrize('k', [1])\n"
        "def test_case(k, monkeypatch): pass\n"
    )
    aliases = AliasTable(tree)
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef))
    assert pytest_parameter_environments(function, aliases) == [{"k": 1}]


def test_literal_evaluator_rejects_unhashable_keys_and_json_type_loss():
    with pytest.raises(ValueError):
        literal_value(ast.parse("{[]: 1}", mode="eval").body, {})
    for expression in ("(1, 2)", "{'shape': (1, 2)}", "{1: 'value'}", "1e1000"):
        with pytest.raises(ValueError):
            json_literal_value(ast.parse(expression, mode="eval").body, {})


@pytest.mark.parametrize(
    "mutation",
    [
        "options['num_ctas'] = 2",
        "options |= {'num_ctas': 2}",
        "options, other = {'num_ctas': 2}, None",
        "options.update({'num_ctas': 2})",
        "mutate(options)",
        "alias = options\nalias['num_ctas'] = 2",
        "if unknown_flag:\n        options = {'num_ctas': 2}",
    ],
)
def test_compile_recipe_rejects_stale_binding_after_mutation(mutation: str):
    source = f"""\
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel(X: tl.constexpr):
    pass

src = ASTSource(fn=kernel, signature={{"X": "constexpr"}}, constexprs={{"X": 1}})
options = {{"num_ctas": 1}}
{mutation}
triton.compile(src, target=GPUTarget("cuda", 90, 32), options=options)
"""
    tree, aliases, candidates = _discover(source)
    assert infer_triton_recipes(candidates, tree, aliases, "case.py") == {}


def test_compile_recipe_preserves_bindings_across_known_compile_calls():
    source = """\
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel(X: tl.constexpr):
    pass

src = ASTSource(fn=kernel, signature={"X": "constexpr"}, constexprs={"X": 1})
triton.compile(src, target=GPUTarget("cuda", 90, 32))
triton.compile(src, target=GPUTarget("cuda", 90, 32))
"""
    tree, aliases, candidates = _discover(source)
    recipes = infer_triton_recipes(candidates, tree, aliases, "case.py")
    assert len(recipes[candidates[0].candidate_id]) == 2


def test_compile_recipe_rejects_unparameterized_fixture_in_payload():
    source = """\
import pytest
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel(X: tl.constexpr):
    pass

@pytest.mark.parametrize("x", [1])
def test_case(x, fixture):
    src = ASTSource(fn=kernel, signature={"X": "constexpr"}, constexprs={"X": x})
    triton.compile(src, target=GPUTarget("cuda", 90, 32), options={"unknown": fixture})
"""
    tree, aliases, candidates = _discover(source)
    assert infer_triton_recipes(candidates, tree, aliases, "case.py") == {}


def test_compile_recipe_rejects_tuple_payload_before_json_serialization():
    source = """\
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel(X: tl.constexpr):
    pass

src = ASTSource(fn=kernel, signature={"X": "constexpr"}, constexprs={"X": (1, 2)})
triton.compile(src, target=GPUTarget("cuda", 90, 32))
"""
    tree, aliases, candidates = _discover(source)
    assert infer_triton_recipes(candidates, tree, aliases, "case.py") == {}


def test_tracked_topk_recipe_ids_and_values():
    recipe_root = ROOT / "corpus" / "upstream" / "triton-61c02bcd" / "recipes" / "20bc89c8423f81e4"
    expected = {
        "20bc89c8423f81e4-sm90-L42-bfe52c49.json": {"K": 1},
        "20bc89c8423f81e4-sm90-L42-661f9844.json": {"K": 8},
    }
    concrete = {path.name: json.loads(path.read_text()) for path in recipe_root.glob("*.json") if path.name != "unresolved.json"}
    assert set(concrete) == set(expected)
    assert {name: recipe["compiler"]["constexprs"] for name, recipe in concrete.items()} == expected



def test_recipe_analysis_classifies_relevant_rejection_gates():
    source = """\
import triton
import triton.language as tl

@triton.jit
def runtime_kernel(pointer, K: tl.constexpr):
    pass

@triton.jit
def grid_kernel(K: tl.constexpr):
    pass

@triton.autotune(configs=[triton.Config({"BLOCK": 32})], key=[])
@triton.jit
def tuned_kernel(BLOCK: tl.constexpr):
    pass

grid = lambda meta: (1,)
runtime_kernel[(1,)](0, K=1)
grid_kernel[grid](K=1)
tuned_kernel[(1,)]()
"""
    tree, aliases, candidates = _discover(source)
    analysis = analyze_triton_recipes(candidates, tree, aliases, "case.py")
    assert analysis.recipes == {}
    assert [attempt["code"] for attempt in analysis.attempts] == [
        "RUNTIME_SPECIALIZATION_UNKNOWN",
        "GRID_UNKNOWN",
        "AUTOTUNE_WINNER_UNKNOWN",
    ]
    assert all(attempt["status"] == "REJECTED" for attempt in analysis.attempts)


def test_recipe_analysis_preserves_accepted_launch_and_target_rejection():
    source = """\
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel(K: tl.constexpr = 4):
    pass

kernel[(1,)](K=1)
src = ASTSource(fn=kernel, signature={"K": "constexpr"}, constexprs={"K": 1})
triton.compile(src, target=GPUTarget("cuda", 80, 32))
"""
    tree, aliases, candidates = _discover(source)
    analysis = analyze_triton_recipes(candidates, tree, aliases, "case.py")
    assert len(analysis.recipes[candidates[0].candidate_id]) == 1
    assert [(attempt["status"], attempt["code"]) for attempt in analysis.attempts] == [
        ("ACCEPTED", "ACCEPTED"),
        ("REJECTED", "TARGET_UNSUPPORTED_OR_UNKNOWN"),
    ]



def test_recipe_analysis_ignores_unrelated_subscript_and_warmup_calls():
    source = """\
import triton

@triton.jit
def kernel():
    pass

factory[0]()
object.warmup(grid=(1,))
"""
    tree, aliases, candidates = _discover(source)
    analysis = analyze_triton_recipes(candidates, tree, aliases, "case.py")
    assert analysis.recipes == {}
    assert analysis.attempts == ()
