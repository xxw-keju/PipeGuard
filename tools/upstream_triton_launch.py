from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Protocol

from upstream_static_eval import closed_value, json_literal_value, keyword_nodes, literal_value, resolve_expression
from upstream_triton_autotune import literal_autotune_configs, literal_node
from upstream_triton_decisions import TritonEvidenceDecision
from upstream_triton_heuristics import apply_literal_heuristics
from upstream_triton_specialization import bind_triton_arguments


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


@dataclass(frozen=True)
class TritonLaunchEvidence:
    candidate: Any
    compiler: dict[str, Any]
    origin_kind: str = "profile-sm90-launch-inference"


_COMPILE_OPTIONS = {"maxnreg", "num_ctas", "num_stages", "num_warps"}


def _closed_grid_value(
    node: ast.AST,
    environment: dict[str, Any],
    aliases: AliasResolver,
    metadata: dict[str, Any],
) -> Any:
    def evaluate_call(function: ast.AST, arguments: list[Any], keywords: dict[str, Any]) -> Any:
        name = aliases.qualified(function)
        if keywords:
            raise ValueError("closed grid call keywords are unsupported")
        if name in {"triton.cdiv", "triton.language.cdiv"} and len(arguments) == 2:
            numerator, denominator = arguments
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in arguments):
                raise ValueError("triton.cdiv arguments are not integers")
            if denominator <= 0:
                raise ValueError("triton.cdiv denominator is not positive")
            return (numerator + denominator - 1) // denominator
        if name == "triton.next_power_of_2" and len(arguments) == 1:
            value = arguments[0]
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError("next_power_of_2 argument is not positive")
            return 1 << (value - 1).bit_length()
        if name in {"min", "max"} and arguments:
            if not all(isinstance(value, int) and not isinstance(value, bool) for value in arguments):
                raise ValueError("closed min/max arguments are not integers")
            return min(arguments) if name == "min" else max(arguments)
        raise ValueError("closed grid call is not whitelisted")

    resolved = resolve_expression(node, environment) or node
    if isinstance(resolved, ast.Lambda):
        arguments = resolved.args
        parameters = (*arguments.posonlyargs, *arguments.args)
        if (
            len(parameters) != 1
            or arguments.vararg is not None
            or arguments.kwarg is not None
            or arguments.kwonlyargs
            or arguments.defaults
        ):
            raise ValueError("closed grid lambda signature is unsupported")
        child = dict(environment)
        child[parameters[0].arg] = metadata
        return closed_value(resolved.body, child, evaluate_call)
    if isinstance(resolved, (ast.FunctionDef, ast.AsyncFunctionDef)):
        arguments = resolved.args
        parameters = (*arguments.posonlyargs, *arguments.args)
        body = [
            statement
            for statement in resolved.body
            if not (
                isinstance(statement, ast.Expr)
                and isinstance(statement.value, ast.Constant)
                and isinstance(statement.value.value, str)
            )
        ]
        if (
            len(parameters) != 1
            or arguments.vararg is not None
            or arguments.kwarg is not None
            or arguments.kwonlyargs
            or arguments.defaults
            or len(body) != 1
            or not isinstance(body[0], ast.Return)
            or body[0].value is None
        ):
            raise ValueError("closed grid function is unsupported")
        child = dict(environment)
        child[parameters[0].arg] = metadata
        return closed_value(body[0].value, child, evaluate_call)
    return closed_value(resolved, environment, evaluate_call)


def _single_cta_grid(
    node: ast.AST,
    environment: dict[str, Any],
    aliases: AliasResolver,
    metadata: dict[str, Any],
) -> list[int] | None:
    try:
        value = literal_value(node, environment)
    except ValueError:
        try:
            value = _closed_grid_value(node, environment, aliases, metadata)
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
    return candidates_by_node.get(id(resolved))


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


def triton_launch_evidences(
    call: ast.Call,
    environment: dict[str, Any],
    aliases: AliasResolver,
    candidates_by_node: dict[int, Any],
) -> tuple[list[TritonLaunchEvidence] | None, str]:
    warmup = isinstance(call.func, ast.Attribute) and call.func.attr == "warmup"
    if warmup:
        candidate_node = call.func.value
        grid_node = None
    elif isinstance(call.func, ast.Subscript):
        candidate_node = call.func.value
        grid_node = call.func.slice
    else:
        return None, "NOT_APPLICABLE"
    candidate = _candidate_from_expression(candidate_node, environment, candidates_by_node)
    if candidate is None:
        return None, "NOT_APPLICABLE"
    keywords = keyword_nodes(call, environment)
    if keywords is None:
        return None, "BINDING_UNKNOWN"
    if warmup:
        grid_node = keywords.get("grid")
        if grid_node is None:
            return None, "GRID_UNKNOWN"
    elif "grid" in keywords:
        return None, "GRID_ARGUMENT_CONFLICT"
    assert grid_node is not None
    if candidate.kind == "AUTOTUNE_JIT":
        configs, code = literal_autotune_configs(candidate, environment, aliases)
        if configs is None:
            return None, code
    elif candidate.kind in {"JIT_ENTRY", "HEURISTICS_JIT"}:
        configs = [None]
    else:
        return None, "CANDIDATE_UNSUPPORTED"
    result: list[TritonLaunchEvidence] = []
    for config in configs:
        merged_keywords = dict(keywords)
        if config is not None:
            if set(config.arguments) & set(merged_keywords):
                return None, "AUTOTUNE_ARGUMENT_CONFLICT"
            if _COMPILE_OPTIONS & set(merged_keywords):
                return None, "AUTOTUNE_OPTION_CONFLICT"
            merged_keywords.update(config.arguments)
            option_nodes = {
                name: literal_node(value) for name, value in config.options.items()
            }
        else:
            option_nodes = {
                name: node for name, node in merged_keywords.items() if name in _COMPILE_OPTIONS
            }
        options = _compiler_options(option_nodes, environment)
        if options is None:
            return None, "OPTIONS_UNSUPPORTED_OR_UNKNOWN"
        ignored = set(option_nodes)
        if warmup:
            ignored.add("grid")
        merged_keywords, code = apply_literal_heuristics(
            candidate,
            call,
            merged_keywords,
            ignored,
            environment,
            aliases,
        )
        if merged_keywords is None:
            return None, code
        binding, code = bind_triton_arguments(
            candidate,
            call,
            environment,
            aliases,
            merged_keywords,
            ignored,
            warmup=warmup,
        )
        if binding is None:
            return None, code
        grid = _single_cta_grid(grid_node, environment, aliases, binding.static_values)
        if grid is None:
            return None, "GRID_UNKNOWN_OR_MULTI_CTA"
        result.append(TritonLaunchEvidence(
            candidate=candidate,
            compiler={
                "signature": binding.signature,
                "constexprs": binding.constexprs,
                "attrs": binding.attrs,
                "options": options,
                "grid_ctas": 1,
                "binding_evidence": None,
            },
            origin_kind=(
                "profile-sm90-autotune-expansion"
                if config is not None else "profile-sm90-launch-inference"
            ),
        ))
    return result, "ACCEPTED"


def triton_launch_evidence(
    call: ast.Call,
    environment: dict[str, Any],
    aliases: AliasResolver,
    candidates_by_node: dict[int, Any],
) -> TritonLaunchEvidence | None:
    evidence, _code = triton_launch_evidences(
        call, environment, aliases, candidates_by_node
    )
    return evidence[0] if evidence else None


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
    accepted, evidence_code = triton_launch_evidences(
        call, environment, aliases, candidates_by_node
    )
    if accepted is not None:
        return TritonEvidenceDecision.accepted_many(mechanism, accepted)
    if candidate.kind == "AUTOTUNE_JIT":
        return TritonEvidenceDecision.rejected(
            mechanism, evidence_code, candidate_id=candidate_id
        )
    if candidate.kind not in {"JIT_ENTRY", "HEURISTICS_JIT"}:
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
    grid = _single_cta_grid(grid_node, environment, aliases, {})
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
