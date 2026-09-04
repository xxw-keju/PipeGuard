from __future__ import annotations

import ast
import re
from dataclasses import dataclass
from typing import Any, Protocol

from upstream_pytest_params import is_pytest_raises_block


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


@dataclass(frozen=True)
class TileLangRecipeClassification:
    kind: str
    reason: str


_UNSUPPORTED_ARCHITECTURE_TOKENS = {
    "amd",
    "blackwell",
    "metal",
    "mfma",
    "rocm",
    "sm100",
    "tcgen",
    "tcgen05",
}


def tilelang_parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    return {
        id(child): parent
        for parent in ast.walk(tree)
        for child in ast.iter_child_nodes(parent)
    }


def _inside_expected_failure(
    candidate: Any,
    aliases: AliasResolver,
    parents: dict[int, ast.AST],
) -> bool:
    node: ast.AST = candidate.node
    while id(node) in parents:
        node = parents[id(node)]
        if isinstance(node, (ast.With, ast.AsyncWith)) and is_pytest_raises_block(node, aliases):
            return True
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and any(
            aliases.qualified(decorator) == "pytest.mark.xfail"
            for decorator in node.decorator_list
        ):
            return True
    return False


def _uses_unsupported_architecture(candidate: Any, aliases: AliasResolver) -> bool:
    for node in ast.walk(candidate.node):
        if not isinstance(node, ast.Call):
            continue
        qualified = aliases.qualified(node.func)
        if qualified is None:
            continue
        tokens = {
            token.lower()
            for token in re.split(r"[^A-Za-z0-9]+|_", qualified)
            if token
        }
        if tokens & _UNSUPPORTED_ARCHITECTURE_TOKENS:
            return True
    return False


def classify_tilelang_recipe(
    candidate: Any,
    aliases: AliasResolver,
    unresolved: set[str],
    parents: dict[int, ast.AST],
) -> TileLangRecipeClassification:
    if candidate.kind != "PRIMFUNC":
        return TileLangRecipeClassification("INSUFFICIENT", "not-primfunc")
    if unresolved:
        return TileLangRecipeClassification("INSUFFICIENT", "unresolved-bindings")
    if _inside_expected_failure(candidate, aliases, parents):
        return TileLangRecipeClassification("HARD_NEGATIVE", "expected-failure-scope")
    if _uses_unsupported_architecture(candidate, aliases):
        return TileLangRecipeClassification("HARD_NEGATIVE", "unsupported-architecture-operation")
    return TileLangRecipeClassification("POSITIVE", "dependency-closed-sm90-profile")
