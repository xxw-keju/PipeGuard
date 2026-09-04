from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Protocol

from upstream_static_eval import json_literal_value, keyword_nodes, literal_value, resolve_expression


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


@dataclass(frozen=True)
class TritonCompileEvidence:
    candidate: Any
    compiler: dict[str, Any]


def _candidate_from_expression(
    node: ast.AST,
    environment: dict[str, Any],
    candidates_by_node: dict[int, Any],
) -> Any | None:
    resolved = resolve_expression(node, environment)
    if not isinstance(resolved, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return candidates_by_node.get(id(resolved))


def _ast_source_payload(
    node: ast.AST,
    environment: dict[str, Any],
    aliases: AliasResolver,
    candidates_by_node: dict[int, Any],
) -> tuple[Any, dict[str, Any]] | None:
    resolved = resolve_expression(node, environment)
    if not isinstance(resolved, ast.Call):
        return None
    if aliases.qualified(resolved.func) not in {
        "triton.compiler.ASTSource",
        "triton.compiler.compiler.ASTSource",
    }:
        return None
    if resolved.args:
        return None
    keywords = keyword_nodes(resolved, environment)
    if keywords is None or not set(keywords) <= {"fn", "signature", "constexprs", "attrs"}:
        return None
    if "fn" not in keywords or "signature" not in keywords:
        return None
    candidate = _candidate_from_expression(keywords["fn"], environment, candidates_by_node)
    if candidate is None:
        return None
    try:
        signature = json_literal_value(keywords["signature"], environment)
        constexprs = json_literal_value(keywords["constexprs"], environment) if "constexprs" in keywords else {}
        attrs = json_literal_value(keywords["attrs"], environment) if "attrs" in keywords else {}
    except ValueError:
        return None
    if not isinstance(signature, dict) or not all(
        isinstance(key, str) and isinstance(value, str) for key, value in signature.items()
    ):
        return None
    if not isinstance(constexprs, dict) or not all(isinstance(key, str) for key in constexprs):
        return None
    if not isinstance(attrs, dict) or not all(isinstance(key, str) for key in attrs):
        return None
    return candidate, {
        "signature": signature,
        "constexprs": constexprs,
        "attrs": attrs,
    }


def _is_exact_sm90_target(node: ast.AST, environment: dict[str, Any], aliases: AliasResolver) -> bool:
    resolved = resolve_expression(node, environment)
    if not isinstance(resolved, ast.Call):
        return False
    if aliases.qualified(resolved.func) != "triton.backends.compiler.GPUTarget":
        return False
    if resolved.keywords or len(resolved.args) != 3:
        return False
    try:
        values = tuple(literal_value(argument, environment) for argument in resolved.args)
    except ValueError:
        return False
    return values == ("cuda", 90, 32)


def triton_compile_evidence(
    call: ast.Call,
    environment: dict[str, Any],
    aliases: AliasResolver,
    candidates_by_node: dict[int, Any],
) -> TritonCompileEvidence | None:
    """Accept only literal ASTSource/options compiled for exact GPUTarget(cuda, 90, 32)."""

    if aliases.qualified(call.func) != "triton.compile":
        return None
    keywords = keyword_nodes(call, environment)
    if keywords is None or not set(keywords) <= {"src", "target", "options"}:
        return None
    if len(call.args) > 1 or (call.args and "src" in keywords):
        return None
    source_node = call.args[0] if call.args else keywords.get("src")
    target_node = keywords.get("target")
    if source_node is None or target_node is None:
        return None
    if not _is_exact_sm90_target(target_node, environment, aliases):
        return None
    source = _ast_source_payload(source_node, environment, aliases, candidates_by_node)
    if source is None:
        return None
    candidate, payload = source
    try:
        options = json_literal_value(keywords["options"], environment) if "options" in keywords else {}
    except ValueError:
        return None
    if not isinstance(options, dict) or not all(isinstance(key, str) for key in options):
        return None
    grid_ctas = options.get("num_ctas", 1)
    if not isinstance(grid_ctas, int) or isinstance(grid_ctas, bool) or grid_ctas < 1:
        return None
    return TritonCompileEvidence(
        candidate=candidate,
        compiler={
            **payload,
            "options": options,
            "grid_ctas": grid_ctas,
            "binding_evidence": None,
        },
    )
