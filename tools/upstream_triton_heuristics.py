from __future__ import annotations

import ast
from typing import Any, Protocol

from upstream_static_eval import closed_value, literal_value, resolve_expression
from upstream_triton_autotune import literal_node


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


def _evaluate_call(
    aliases: AliasResolver,
    function: ast.AST,
    arguments: list[Any],
    keywords: dict[str, Any],
) -> Any:
    if keywords:
        raise ValueError("heuristic call keywords are unsupported")
    name = aliases.qualified(function)
    if name in {"triton.cdiv", "triton.language.cdiv"} and len(arguments) == 2:
        left, right = arguments
        if not all(isinstance(value, int) and not isinstance(value, bool) for value in arguments) or right <= 0:
            raise ValueError("heuristic cdiv arguments are invalid")
        return (left + right - 1) // right
    if name == "triton.next_power_of_2" and len(arguments) == 1:
        value = arguments[0]
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("heuristic next_power_of_2 argument is invalid")
        return 1 << (value - 1).bit_length()
    if name in {"min", "max"} and arguments and all(
        isinstance(value, int) and not isinstance(value, bool) for value in arguments
    ):
        return min(arguments) if name == "min" else max(arguments)
    raise ValueError("heuristic call is not whitelisted")


def _known_arguments(
    candidate: Any,
    call: ast.Call,
    keywords: dict[str, ast.AST],
    ignored_keywords: set[str],
    environment: dict[str, Any],
) -> dict[str, Any]:
    arguments = candidate.node.args
    positional = (*arguments.posonlyargs, *arguments.args)
    parameters = (*positional, *arguments.kwonlyargs)
    nodes = {parameter.arg: value for parameter, value in zip(positional, call.args)}
    nodes.update({
        name: value for name, value in keywords.items()
        if name not in ignored_keywords and name in {parameter.arg for parameter in parameters}
    })
    if arguments.defaults:
        for parameter, value in zip(positional[-len(arguments.defaults):], arguments.defaults):
            nodes.setdefault(parameter.arg, value)
    for parameter, value in zip(arguments.kwonlyargs, arguments.kw_defaults):
        if value is not None:
            nodes.setdefault(parameter.arg, value)
    result: dict[str, Any] = {}
    for name, node in nodes.items():
        try:
            result[name] = literal_value(node, environment)
        except ValueError:
            continue
    return result


def apply_literal_heuristics(
    candidate: Any,
    call: ast.Call,
    keywords: dict[str, ast.AST],
    ignored_keywords: set[str],
    environment: dict[str, Any],
    aliases: AliasResolver,
) -> tuple[dict[str, ast.AST] | None, str]:
    result = dict(keywords)
    decorators = [
        decorator for decorator in candidate.node.decorator_list
        if isinstance(decorator, ast.Call) and aliases.qualified(decorator.func) == "triton.heuristics"
    ]
    for decorator in decorators:
        if (
            len(decorator.args) > 1
            or any(keyword.arg is None for keyword in decorator.keywords)
            or any(keyword.arg != "values" for keyword in decorator.keywords)
        ):
            return None, "HEURISTIC_VALUES_UNKNOWN"
        values_nodes = [keyword.value for keyword in decorator.keywords if keyword.arg == "values"]
        if decorator.args and values_nodes or len(values_nodes) > 1:
            return None, "HEURISTIC_VALUES_UNKNOWN"
        values_node = decorator.args[0] if decorator.args else (values_nodes[0] if values_nodes else None)
        resolved = resolve_expression(values_node, environment) if values_node is not None else None
        resolved = resolved or values_node
        if not isinstance(resolved, ast.Dict) or any(key is None for key in resolved.keys):
            return None, "HEURISTIC_VALUES_UNKNOWN"
        seen_keys: set[str] = set()
        for key_node, function_node in zip(resolved.keys, resolved.values):
            try:
                key = literal_value(key_node, environment)
            except ValueError:
                return None, "HEURISTIC_VALUES_UNKNOWN"
            function = resolve_expression(function_node, environment) or function_node
            if (
                not isinstance(key, str)
                or key in seen_keys
                or not isinstance(function, ast.Lambda)
            ):
                return None, "HEURISTIC_VALUES_UNKNOWN"
            seen_keys.add(key)
            args = function.args
            parameters = (*args.posonlyargs, *args.args)
            if (
                len(parameters) != 1
                or args.vararg is not None
                or args.kwarg is not None
                or args.kwonlyargs
                or args.defaults
            ):
                return None, "HEURISTIC_VALUES_UNKNOWN"
            metadata = _known_arguments(
                candidate, call, result, ignored_keywords, environment
            )
            child = dict(environment)
            child[parameters[0].arg] = metadata
            try:
                value = closed_value(
                    function.body,
                    child,
                    lambda fn, positional, named: _evaluate_call(
                        aliases, fn, positional, named
                    ),
                )
                result[key] = literal_node(value)
            except ValueError:
                return None, "HEURISTIC_VALUE_UNKNOWN"
    return result, "ACCEPTED"
