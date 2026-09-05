from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import jsonschema

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from upstream_extract import (  # noqa: E402
    AliasTable,
    CandidateVisitor,
    candidate_roles,
    dependency_nodes,
    infer_triton_recipes,
    make_capsule,
)


def _discover(dsl: str, source: str):
    tree = ast.parse(source)
    visitor = CandidateVisitor(dsl, "case.py", source, AliasTable(tree))
    visitor.visit(tree)
    return tree, visitor.candidates


def test_tilelang_aliases_nested_primfunc_and_jit_are_discovered():
    source = """\
import tilelang as tl
from tilelang import language as T

@tl.jit
def factory(size):
    @T.prim_func
    def kernel(a: T.handle):
        T.evaluate(size)
    return kernel
"""
    _tree, candidates = _discover("tilelang", source)
    assert [(item.qualname, item.kind) for item in candidates] == [
        ("factory", "JIT_ENTRY"),
        ("factory.kernel", "PRIMFUNC"),
    ]


def test_tirx_and_legacy_tilelang_names_are_discovered():
    source = """\
from tvm.script import tirx
from tvm.tl import language as T

@tirx.prim_func
def first():
    pass

@T.prim_func
def second():
    pass
"""
    _tree, candidates = _discover("tilelang", source)
    assert [item.qualname for item in candidates] == ["first", "second"]


def test_triton_decorator_stack_and_local_kernel_are_discovered():
    source = """\
import triton
import triton.language as tl

@triton.autotune(configs=[], key=[])
@triton.jit
def root(x, BLOCK: tl.constexpr):
    return

def test_case(dtype):
    @triton.jit
    def local(x):
        return
    local[(1,)](dtype)
"""
    tree, candidates = _discover("triton", source)
    roles, _dependencies = candidate_roles(candidates, tree)
    assert [(item.qualname, item.kind) for item in candidates] == [
        ("root", "AUTOTUNE_JIT"),
        ("test_case.local", "JIT_ENTRY"),
    ]
    assert roles[candidates[1].candidate_id] == "ROOT"


def test_nested_capsule_is_parseable_and_exposes_resolver():
    source = """\
import triton

def test_case(value):
    scale = value + 1
    @triton.jit
    def kernel(x):
        return x + scale
    kernel[(1,)](value)
"""
    tree, candidates = _discover("triton", source)
    candidate = candidates[0]
    dependencies, unresolved = dependency_nodes(candidate, tree)
    capsule = make_capsule(candidate, dependencies, unresolved, source.splitlines(keepends=True))
    module = ast.parse(capsule)
    assert any(isinstance(node, ast.FunctionDef) and node.name == "pipeguard_resolve" for node in module.body)
    assert "pipeguard_bindings['scale']" in capsule


def test_dependency_nodes_exclude_helper_and_class_parameters():
    source = """import triton
import triton.language as tl

@triton.jit
def helper(x):
    return x

@triton.aggregate
class Box:
    value: tl.tensor

    def __init__(self, x):
        self.value = x

def test_case():
    @triton.jit
    def kernel():
        value = helper(1)
        box = Box(value)
"""
    tree, candidates = _discover("triton", source)
    candidate = next(item for item in candidates if item.qualname == "test_case.kernel")
    dependencies, unresolved = dependency_nodes(candidate, tree)
    assert unresolved == set()
    assert [node.name for node in dependencies if isinstance(node, (ast.FunctionDef, ast.ClassDef))] == [
        "helper",
        "Box",
    ]


def test_triton_recipe_requires_literal_astsource_and_explicit_sm90_target():
    source = """\
import triton
import triton.language as tl
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

def test_case():
    @triton.jit
    def kernel(src, dst):
        tl.store(dst, tl.load(src))
    source = ASTSource(fn=kernel, signature={"src": "*fp32", "dst": "*fp32"})
    triton.compile(source, target=GPUTarget("cuda", 90, 32))
    triton.compile(source, target=GPUTarget("cuda", 80, 32))
"""
    tree, candidates = _discover("triton", source)
    recipes = infer_triton_recipes(candidates, tree, AliasTable(tree), "case.py")
    assert list(recipes) == [candidates[0].candidate_id]
    recipe = recipes[candidates[0].candidate_id][0]
    assert recipe["origin"] == "static-inference:case.py:11-11"
    assert recipe["compiler"] == {
        "signature": {"src": "*fp32", "dst": "*fp32"},
        "constexprs": {},
        "attrs": {},
        "options": {},
        "grid_ctas": 1,
        "binding_evidence": None,
    }


def test_triton_recipe_rejects_dynamic_values_and_expected_failure_control_flow():
    source = """\
import pytest
import triton
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource

@triton.jit
def kernel(x, BLOCK):
    return

def test_case(block):
    dynamic = ASTSource(fn=kernel, signature={"x": "*fp32", "BLOCK": "constexpr"}, constexprs={"BLOCK": block})
    triton.compile(dynamic, target=GPUTarget("cuda", 90, 32))
    literal = ASTSource(fn=kernel, signature={"x": "*fp32", "BLOCK": "constexpr"}, constexprs={"BLOCK": 128})
    with pytest.raises(Exception):
        triton.compile(literal, target=GPUTarget("cuda", 90, 32))
"""
    tree, candidates = _discover("triton", source)
    assert infer_triton_recipes(candidates, tree, AliasTable(tree), "case.py") == {}


def test_locked_source_and_generated_contracts_validate():
    schema_root = ROOT / "corpus" / "upstream" / "schemas"
    source_root = ROOT / "corpus" / "upstream"
    source_schema = json.loads((schema_root / "sources-v1.schema.json").read_text())
    jsonschema.validate(json.loads((source_root / "sources.json").read_text()), source_schema)
    for corpus_name in ("tilelang-2c31e071", "triton-61c02bcd"):
        index_path = source_root / corpus_name / "source-index.json"
        if not index_path.exists():
            continue
        index = json.loads(index_path.read_text())
        schema = json.loads((schema_root / "source-index-v1.schema.json").read_text())
        jsonschema.validate(index, schema)
        candidate_ids = [item["candidate_id"] for item in index["candidates"]]
        assert len(candidate_ids) == len(set(candidate_ids))
        for item in index["candidates"]:
            capsule = source_root / item["capsule_path"]
            manifest = source_root / item["manifest_path"]
            ast.parse(capsule.read_text(), filename=str(capsule))
            assert manifest.exists()


def test_extractor_cli_rejects_unlocked_checkout(tmp_path: Path):
    source = tmp_path / "source"
    source.mkdir()
    subprocess.run(["git", "init", "-q", str(source)], check=True)
    (source / "examples").mkdir()
    (source / "examples" / "case.py").write_text("import triton\n@triton.jit\ndef kernel(x):\n    return x\n")
    subprocess.run(["git", "-C", str(source), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(source), "-c", "user.name=PipeGuard", "-c", "user.email=pipeguard@example.invalid", "commit", "-qm", "fixture"],
        check=True,
    )
    process = subprocess.run(
        [
            sys.executable,
            str(ROOT / "tools" / "upstream_extract.py"),
            "--dsl", "triton",
            "--triton-root", str(source),
            "--output", str(tmp_path / "out"),
        ],
        text=True,
        capture_output=True,
    )
    assert process.returncode != 0
    assert "checkout mismatch" in process.stderr
