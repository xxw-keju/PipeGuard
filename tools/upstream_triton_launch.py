from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Protocol

from upstream_static_eval import json_literal_value, keyword_nodes, literal_value, resolve_expression
from upstream_triton_decisions import TritonEvidenceDecision
from upstream_triton_specialization import bind_triton_arguments


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


@dataclass(frozen=True)
class TritonLaunchEvidence:
    candidate: Any
    compiler: dict[str, Any]
    origin_kind: str = "profile-sm90-launch-inference"


_COMPILE_OPTIONS = {"maxnreg", "num_ctas", "num_stages", "num_warps"}


def _is_constexpr(annotation: ast.AST | None, aliases: AliasResolver) -> bool:
    if annotation is None:
        return False
    if aliases.qualified(annotation) == "triton.language.constexpr":
        return True
    return isinstance(annotation, ast.Constant) and isinstance(annotation.value, str) and annotation.value in {
        "constexpr",
        "tl.constexpr",
        "triton.language.constexpr",
    }


def _single_cta_grid(node: ast.AST, environment: dict[str, Any]) -> list[int] | None:
    try:
        value = literal_value(node, environment)
    except ValueError:
        return None
    if value == 1 and not isinstance(value, bool):
        return [1]
    if (
        isinstance(value, (list, tuple))
        and 1 <= len(value) <= 3
        and all(item == 1 and not isinstance(item, bool) for item in value)
    ):
        return [1 for _item in value]
    return None


def _candidate_from_expression(
    node: ast.AST,
    environment: dict[str, Any],
    candidates_by_node: dict[int, Any],
) -> Any | None:
    resolved = resolve_expression(node, environment)
    if not isinstance(resolved, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    candidate = candidates_by_node.get(id(resolved))
    if candidate is None or candidate.kind != "JIT_ENTRY":
        return None
    return candidate


def _parameter_defaults(arguments: ast.arguments) -> dict[str, ast.AST]:
    positional = (*arguments.posonlyargs, *arguments.args)
    defaults = {
        parameter.arg: value
        for parameter, value in zip(positional[-len(arguments.defaults):], arguments.defaults)
    } if arguments.defaults else {}
    defaults.update({
        parameter.arg: value
        for parameter, value in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if value is not None
    })
    return defaults


def _compiler_options(nodes: dict[str, ast.AST], environment: dict[str, Any]) -> dict[str, Any] | None:
    try:
        options = {name: json_literal_value(node, environment) for name, node in nodes.items()}
    except ValueError:
        return None
    num_warps = options.get("num_warps", 4)
    num_stages = options.get("num_stages", 3)
    num_ctas = options.get("num_ctas", 1)
    maxnreg = options.get("maxnreg")
    if (
        not isinstance(num_warps, int)
        or isinstance(num_warps, bool)
        or num_warps < 1
        or num_warps & (num_warps - 1)
        or not isinstance(num_stages, int)
        or isinstance(num_stages, bool)
        or num_stages < 1
        or num_ctas != 1
        or isinstance(num_ctas, bool)
        or (maxnreg is not None and (
            not isinstance(maxnreg, int) or isinstance(maxnreg, bool) or maxnreg < 1
        ))
    ):
        return None
    return options


def _bind_constexpr_arguments(
    candidate: Any,
    call: ast.Call,
    environment: dict[str, Any],
    aliases: AliasResolver,
    ignored_keywords: set[str],
) -> tuple[dict[str, str], dict[str, Any]] | None:
    arguments = candidate.node.args
    if arguments.vararg is not None or arguments.kwarg is not None:
        return None
    positional = (*arguments.posonlyargs, *arguments.args)
    parameters = (*positional, *arguments.kwonlyargs)
    if any(not _is_constexpr(parameter.annotation, aliases) for parameter in parameters):
        return None
    if any(isinstance(value, ast.Starred) for value in call.args) or len(call.args) > len(positional):
        return None
    bound: dict[str, ast.AST] = {
        parameter.arg: value for parameter, value in zip(positional, call.args)
    }
    for keyword in call.keywords:
        if keyword.arg is None:
            return None
        if keyword.arg in ignored_keywords:
            continue
        if keyword.arg not in {parameter.arg for parameter in parameters} or keyword.arg in bound:
            return None
        if keyword.arg in {parameter.arg for parameter in arguments.posonlyargs}:
            return None
        bound[keyword.arg] = keyword.value
    defaults = _parameter_defaults(arguments)
    try:
        constexprs = {
            parameter.arg: json_literal_value(bound.get(parameter.arg, defaults[parameter.arg]), environment)
            for parameter in parameters
        }
    except (KeyError, ValueError):
        return None
    return ({parameter.arg: "constexpr" for parameter in parameters}, constexprs)


def triton_launch_evidence(
    call: ast.Call,
    environment: dict[str, Any],
    aliases: AliasResolver,
    candidates_by_node: dict[int, Any],
) -> TritonLaunchEvidence | None:
    """Infer compile inputs only for profile-targeted, all-constexpr single-CTA calls."""

    warmup = isinstance(call.func, ast.Attribute) and call.func.attr == "warmup"
    if warmup:
        candidate_node = call.func.value
        grid_node = None
    elif isinstance(call.func, ast.Subscript):
        candidate_node = call.func.value
        grid_node = call.func.slice
    else:
        return None
    candidate = _candidate_from_expression(candidate_node, environment, candidates_by_node)
    if candidate is None:
        return None
    keywords = keyword_nodes(call, environment)
    if keywords is None:
        return None
    if warmup:
        grid_node = keywords.get("grid")
        if grid_node is None:
            return None
    elif "grid" in keywords:
        return None
    assert grid_node is not None
    grid = _single_cta_grid(grid_node, environment)
    if grid is None:
        return None
    option_nodes = {name: node for name, node in keywords.items() if name in _COMPILE_OPTIONS}
    options = _compiler_options(option_nodes, environment)
    if options is None:
        return None
    ignored = set(option_nodes)
    if warmup:
        ignored.add("grid")
    binding, _code = bind_triton_arguments(
        candidate,
        call,
        environment,
        aliases,
        keywords,
        ignored,
        warmup=warmup,
    )
    if binding is None:
        return None
    return TritonLaunchEvidence(
        candidate=candidate,
        compiler={
            "signature": binding.signature,
            "constexprs": binding.constexprs,
            "attrs": binding.attrs,
            "options": options,
            "grid_ctas": 1,
            "binding_evidence": None,
        },
    )


def _resolved_launch_candidate(
    node: ast.AST,
    environment: dict[str, Any],
    candidates_by_node: dict[int, Any],
) -> Any | None:
    resolved = resolve_expression(node, environment)
    if not isinstance(resolved, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return None
    return candidates_by_node.get(id(resolved))


def analyze_triton_launch(
    call: ast.Call,
    environment: dict[str, Any],
    aliases: AliasResolver,
    candidates_by_node: dict[int, Any],
) -> TritonEvidenceDecision:
    """Classify one launch/warmup call without changing evidence acceptance."""

    warmup = isinstance(call.func, ast.Attribute) and call.func.attr == "warmup"
    bracket = isinstance(call.func, ast.Subscript)
    mechanism = "warmup" if warmup else "launch"
    if not warmup and not bracket:
        return TritonEvidenceDecision.not_applicable(mechanism)
    candidate_node = call.func.value
    candidate = _resolved_launch_candidate(candidate_node, environment, candidates_by_node)
    if candidate is None:
        return TritonEvidenceDecision.not_applicable(mechanism)
    candidate_id = candidate.candidate_id
    if candidate.kind == "AUTOTUNE_JIT":
        return TritonEvidenceDecision.rejected(
            mechanism, "AUTOTUNE_WINNER_UNKNOWN", candidate_id=candidate_id
        )
    if candidate.kind != "JIT_ENTRY":
        return TritonEvidenceDecision.rejected(
            mechanism, "CANDIDATE_UNSUPPORTED", candidate_id=candidate_id
        )
    keywords = keyword_nodes(call, environment)
    if keywords is None:
        return TritonEvidenceDecision.rejected(
            mechanism, "BINDING_UNKNOWN", candidate_id=candidate_id
        )
    grid_node = keywords.get("grid") if warmup else call.func.slice
    if warmup and grid_node is None:
        return TritonEvidenceDecision.rejected(
            mechanism, "GRID_UNKNOWN", candidate_id=candidate_id
        )
    if not warmup and "grid" in keywords:
        return TritonEvidenceDecision.rejected(
            mechanism, "GRID_ARGUMENT_CONFLICT", candidate_id=candidate_id
        )
    assert grid_node is not None
    grid = _single_cta_grid(grid_node, environment)
    if grid is None:
        try:
            literal_value(grid_node, environment)
        except ValueError:
            code = "GRID_UNKNOWN"
        else:
            code = "GRID_MULTI_CTA_OR_UNSUPPORTED"
        return TritonEvidenceDecision.rejected(mechanism, code, candidate_id=candidate_id)
    option_nodes = {name: node for name, node in keywords.items() if name in _COMPILE_OPTIONS}
    if option_nodes:
        try:
            for node in option_nodes.values():
                json_literal_value(node, environment)
        except ValueError:
            return TritonEvidenceDecision.rejected(
                mechanism, "OPTIONS_UNKNOWN", candidate_id=candidate_id
            )
    if _compiler_options(option_nodes, environment) is None:
        return TritonEvidenceDecision.rejected(
            mechanism, "OPTIONS_UNSUPPORTED", candidate_id=candidate_id
        )
    ignored = set(option_nodes)
    if warmup:
        ignored.add("grid")
    binding, code = bind_triton_arguments(
        candidate,
        call,
        environment,
        aliases,
        keywords,
        ignored,
        warmup=warmup,
    )
    if binding is None:
        return TritonEvidenceDecision.rejected(
            mechanism, code, candidate_id=candidate_id
        )
    evidence = triton_launch_evidence(call, environment, aliases, candidates_by_node)
    if evidence is None:
        return TritonEvidenceDecision.rejected(
            mechanism, "SPECIALIZATION_UNKNOWN", candidate_id=candidate_id
        )
    return TritonEvidenceDecision.accepted(mechanism, evidence)
