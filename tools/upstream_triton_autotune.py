from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Protocol

from upstream_static_eval import json_literal_value, keyword_nodes, literal_value, resolve_expression


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


@dataclass(frozen=True)
class TritonConfigEvidence:
    arguments: dict[str, ast.AST]
    options: dict[str, Any]


def literal_node(value: Any) -> ast.AST:
    if isinstance(value, (str, int, float, bool, type(None))):
        return ast.Constant(value=value)
    if isinstance(value, list):
        return ast.List(elts=[literal_node(item) for item in value], ctx=ast.Load())
    if isinstance(value, dict) and all(isinstance(key, str) for key in value):
        return ast.Dict(
            keys=[ast.Constant(value=key) for key in value],
            values=[literal_node(item) for item in value.values()],
        )
    raise ValueError("autotune value cannot be represented as literal AST")


def _none_or_missing(nodes: dict[str, ast.AST], name: str, environment: dict[str, Any]) -> bool:
    if name not in nodes:
        return True
    try:
        return literal_value(nodes[name], environment) is None
    except ValueError:
        return False


def _config(
    node: ast.AST,
    environment: dict[str, Any],
    aliases: AliasResolver,
) -> TritonConfigEvidence | None:
    resolved = resolve_expression(node, environment) or node
    if not isinstance(resolved, ast.Call) or aliases.qualified(resolved.func) != "triton.Config":
        return None
    if len(resolved.args) > 1:
        return None
    nodes = keyword_nodes(resolved, environment)
    if nodes is None or not set(nodes) <= {
        "kwargs", "num_warps", "num_stages", "num_ctas", "maxnreg", "pre_hook", "ir_override"
    }:
        return None
    if resolved.args and "kwargs" in nodes:
        return None
    kwargs_node = resolved.args[0] if resolved.args else nodes.get("kwargs")
    if kwargs_node is None or not _none_or_missing(nodes, "pre_hook", environment) or not _none_or_missing(
        nodes, "ir_override", environment
    ):
        return None
    try:
        arguments = json_literal_value(kwargs_node, environment)
        options = {
            name: json_literal_value(nodes[name], environment) if name in nodes else default
            for name, default in (
                ("num_warps", 4),
                ("num_stages", 3),
                ("num_ctas", 1),
                ("maxnreg", None),
            )
        }
    except ValueError:
        return None
    if not isinstance(arguments, dict) or not all(isinstance(name, str) for name in arguments):
        return None
    num_warps = options["num_warps"]
    num_stages = options["num_stages"]
    num_ctas = options["num_ctas"]
    maxnreg = options["maxnreg"]
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
    try:
        argument_nodes = {name: literal_node(value) for name, value in arguments.items()}
    except ValueError:
        return None
    return TritonConfigEvidence(argument_nodes, options)


def literal_autotune_configs(
    candidate: Any,
    environment: dict[str, Any],
    aliases: AliasResolver,
) -> tuple[list[TritonConfigEvidence] | None, str]:
    decorators = [
        decorator
        for decorator in candidate.node.decorator_list
        if isinstance(decorator, ast.Call) and aliases.qualified(decorator.func) == "triton.autotune"
    ]
    if len(decorators) != 1:
        return None, "AUTOTUNE_DECORATOR_UNKNOWN"
    decorator = decorators[0]
    if len(decorator.args) > 2:
        return None, "AUTOTUNE_DECORATOR_UNKNOWN"
    nodes = keyword_nodes(decorator, environment)
    if nodes is None or not set(nodes) <= {
        "configs", "key", "prune_configs_by", "reset_to_zero", "restore_value",
        "pre_hook", "post_hook", "warmup", "rep", "use_cuda_graph", "do_bench",
        "cache_results",
    }:
        return None, "AUTOTUNE_DECORATOR_UNKNOWN"
    positional_names = ("configs", "key")
    for name, value in zip(positional_names, decorator.args):
        if name in nodes:
            return None, "AUTOTUNE_DECORATOR_UNKNOWN"
        nodes[name] = value
    if "configs" not in nodes or "key" not in nodes:
        return None, "AUTOTUNE_DECORATOR_UNKNOWN"
    for effect_name in (
        "prune_configs_by", "reset_to_zero", "restore_value", "pre_hook", "post_hook"
    ):
        if not _none_or_missing(nodes, effect_name, environment):
            return None, "AUTOTUNE_DYNAMIC_EFFECT"
    resolved = resolve_expression(nodes["configs"], environment) or nodes["configs"]
    if not isinstance(resolved, (ast.List, ast.Tuple)) or not resolved.elts:
        return None, "AUTOTUNE_CONFIGS_UNKNOWN"
    configs: list[TritonConfigEvidence] = []
    for item in resolved.elts:
        config = _config(item, environment, aliases)
        if config is None:
            return None, "AUTOTUNE_CONFIGS_UNKNOWN"
        configs.append(config)
    return configs, "ACCEPTED"
