from __future__ import annotations

import ast
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from upstream_extract import AliasTable, CandidateVisitor, dependency_nodes  # noqa: E402
from upstream_tilelang_recipes import classify_tilelang_recipe, tilelang_parent_map  # noqa: E402


SOURCE = """import pytest
import tilelang
from tilelang import language as T

@T.prim_func
def positive(A: T.Buffer((1,), "float32")):
    A[0] = 1

with pytest.raises(ValueError):
    @T.prim_func
    def negative(A: T.Buffer((1,), "float32")):
        A[0] = 1

@pytest.mark.xfail
def test_expected_failure():
    @T.prim_func
    def xfail_kernel(A: T.Buffer((1,), "float32")):
        A[0] = 1

@T.prim_func
def blackwell_kernel():
    T.tcgen05_mma()

@T.prim_func
def unresolved_kernel():
    value = UNKNOWN_VALUE

@tilelang.jit
def factory():
    return positive
"""


def _classifications(relative_path: str = "testing/python/cuda/test_case.py"):
    tree = ast.parse(SOURCE)
    aliases = AliasTable(tree)
    visitor = CandidateVisitor("tilelang", relative_path, SOURCE, aliases)
    visitor.visit(tree)
    parents = tilelang_parent_map(tree)
    return {
        candidate.name: classify_tilelang_recipe(
            candidate,
            aliases,
            dependency_nodes(candidate, tree)[1],
            parents,
        )
        for candidate in visitor.candidates
    }


def test_tilelang_recipe_classification_separates_positive_negative_and_insufficient():
    classifications = _classifications()
    assert (classifications["positive"].kind, classifications["positive"].reason) == (
        "POSITIVE",
        "dependency-closed-sm90-profile",
    )
    assert classifications["negative"].reason == "expected-failure-scope"
    assert classifications["negative"].kind == "HARD_NEGATIVE"
    assert classifications["xfail_kernel"].kind == "HARD_NEGATIVE"
    assert classifications["blackwell_kernel"].reason == "unsupported-architecture-operation"
    assert classifications["blackwell_kernel"].kind == "HARD_NEGATIVE"
    assert classifications["unresolved_kernel"].kind == "INSUFFICIENT"
    assert classifications["factory"].reason == "not-primfunc"
    assert classifications["factory"].kind == "INSUFFICIENT"


def test_source_directory_does_not_substitute_for_architecture_operation_evidence():
    for relative_path in (
        "testing/python/metal/test_generic_kernel.py",
        "testing/python/amd/test_generic_kernel.py",
        "testing/python/cpu/test_generic_kernel.py",
    ):
        classifications = _classifications(relative_path)
        assert classifications["positive"].kind == "POSITIVE"
