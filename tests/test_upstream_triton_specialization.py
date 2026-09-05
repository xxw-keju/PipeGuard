from __future__ import annotations

import ast
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from upstream_extract import AliasTable, CandidateVisitor, infer_triton_recipes  # noqa: E402
from upstream_worker import _triton_attrs, _triton_compiler_constexprs, _triton_resolver_kwargs  # noqa: E402


def _recipes(source: str) -> list[dict]:
    tree = ast.parse(source)
    aliases = AliasTable(tree)
    visitor = CandidateVisitor("triton", "specialization.py", source, aliases)
    visitor.visit(tree)
    inferred = infer_triton_recipes(visitor.candidates, tree, aliases, "specialization.py")
    return [recipe for recipes in inferred.values() for recipe in recipes]


def _compiler(source: str) -> dict:
    recipes = _recipes(source)
    assert len(recipes) == 1
    return recipes[0]["compiler"]


def test_scalar_specialization_matches_locked_native_boundaries():
    compiler = _compiler('''
import triton

@triton.jit
def kernel(zero, one, flag, ratio, large):
    pass

kernel[(1,)](0, 1, True, 1.5, 2147483648)
''')
    assert compiler["signature"] == {
        "zero": "i32",
        "one": "constexpr",
        "flag": "u1",
        "ratio": "fp32",
        "large": "i64",
    }
    assert compiler["constexprs"] == {"one": 1}
    assert compiler["attrs"] == {
        "zero": [["tt.divisibility", 16]],
        "large": [["tt.divisibility", 16]],
    }


def test_specialization_flags_match_locked_native_attrs():
    compiler = _compiler('''
import triton

@triton.jit(
    do_not_specialize=["plain"],
    do_not_specialize_on_alignment=["unaligned"],
)
def kernel(plain, unaligned, normal):
    pass

kernel[(1,)](16, 16, 17)
''')
    assert compiler["signature"] == {"plain": "i32", "unaligned": "i32", "normal": "i32"}
    assert compiler["constexprs"] == {}
    assert compiler["attrs"] == {"unaligned": [], "normal": []}


def test_dtype_warmup_reconstructs_mock_tensor_pointer_attrs():
    compiler = _compiler('''
import torch
import triton

@triton.jit
def kernel(pointer, size):
    pass

kernel.warmup(torch.float32, 32, grid=(1,))
''')
    assert compiler["signature"] == {"pointer": "*fp32", "size": "i32"}
    assert compiler["constexprs"] == {}
    assert compiler["attrs"] == {
        "pointer": [["tt.divisibility", 16]],
        "size": [["tt.divisibility", 16]],
    }


def test_warmup_does_not_wrap_keyword_dtype_and_unknown_tensor_factory():
    assert _recipes('''
import torch
import triton

@triton.jit
def kernel(pointer):
    pass

kernel.warmup(pointer=torch.float32, grid=(1,))
''') == []
    assert _recipes('''
import torch
import triton

@triton.jit
def kernel(pointer):
    pass

kernel[(1,)](torch.empty(1))
''') == []


def test_explicit_constexpr_without_default_and_scalar_annotation_are_supported():
    compiler = _compiler('''
import triton
import triton.language as tl

@triton.jit
def kernel(value: "int64", BLOCK: tl.constexpr):
    pass

kernel[(1,)](17, BLOCK=64)
''')
    assert compiler["signature"] == {"value": "i64", "BLOCK": "constexpr"}
    assert compiler["constexprs"] == {"BLOCK": 64}
    assert compiler["attrs"] == {"value": []}


def test_out_of_range_integer_and_unknown_annotation_fail_closed():
    assert _recipes('''
import triton

@triton.jit
def kernel(value):
    pass

kernel[(1,)](18446744073709551616)
''') == []
    assert _recipes('''
import triton

@triton.jit
def kernel(value: UnknownType):
    pass

kernel[(1,)](1)
''') == []


def test_worker_rehydrates_parameter_attrs_to_native_paths():
    function = type("Function", (), {"arg_names": ["left", "right"]})()
    assert _triton_attrs(function, {
        "left": [["tt.divisibility", 16]],
        "right": [],
    }) == {
        (0,): [["tt.divisibility", 16]],
        (1,): [],
    }
    with pytest.raises(ValueError, match="unknown Triton attribute parameter"):
        _triton_attrs(function, {"missing": []})
    with pytest.raises(TypeError, match="invalid Triton attributes"):
        _triton_attrs(function, {"left": ["tt.divisibility", 16]})


def test_worker_materializes_explicit_triton_constexpr_resolver_values():
    constructor = lambda value: ("constexpr", value)
    assert _triton_resolver_kwargs({
        "plain": 7,
        "closed": {"__pipeguard_triton_constexpr__": 42},
    }, constructor) == {
        "plain": 7,
        "closed": ("constexpr", 42),
    }
    with pytest.raises(ValueError, match="malformed Triton constexpr"):
        _triton_resolver_kwargs({
            "closed": {"__pipeguard_triton_constexpr__": 42, "extra": True},
        }, constructor)
    with pytest.raises(TypeError, match="unsupported Triton constexpr"):
        _triton_resolver_kwargs({
            "closed": {"__pipeguard_triton_constexpr__": [42]},
        }, constructor)


def test_worker_materializes_explicit_triton_dtype_constexpr_values():
    constructor = lambda value: ("dtype", value)
    assert _triton_compiler_constexprs({
        "plain": 7,
        "dtype": {"__pipeguard_triton_dtype__": "int32"},
    }, constructor) == {
        "plain": 7,
        "dtype": ("dtype", "int32"),
    }
    with pytest.raises(ValueError, match="malformed Triton dtype"):
        _triton_compiler_constexprs({
            "dtype": {"__pipeguard_triton_dtype__": "int32", "extra": True},
        }, constructor)
    with pytest.raises(TypeError, match="unsupported Triton dtype"):
        _triton_compiler_constexprs({
            "dtype": {"__pipeguard_triton_dtype__": 32},
        }, constructor)
