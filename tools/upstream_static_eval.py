from __future__ import annotations

import ast
from typing import Any


AMBIGUOUS = object()
_MISSING = object()
_LITERAL_SCALARS = (str, int, float, bool, type(None))


def bind_once(environment: dict[str, Any], name: str, value: Any) -> None:
    """Bind one syntax node, marking conflicting evidence as ambiguous."""

    previous = environment.get(name, _MISSING)
    if previous is _MISSING or previous is value:
        environment[name] = value
    else:
        environment[name] = AMBIGUOUS


def resolve_expression(node: ast.AST, environment: dict[str, Any]) -> ast.AST | None:
    """Resolve aliases through the static environment without evaluating code."""

    seen: set[str] = set()
    while isinstance(node, ast.Name):
        if node.id in seen:
            return None
        seen.add(node.id)
        value = environment.get(node.id, _MISSING)
        if value is _MISSING or value is AMBIGUOUS or not isinstance(value, ast.AST):
            return None
        node = value
    return node


def _bound_value(name: str, environment: dict[str, Any], seen: set[str]) -> Any:
    if name in seen:
        raise ValueError("cyclic static binding")
    value = environment.get(name, _MISSING)
    if value is _MISSING or value is AMBIGUOUS:
        raise ValueError("unresolved literal")
    if isinstance(value, ast.AST):
        return _literal_value(value, environment, seen | {name})
    return _copy_literal(value)


def _copy_literal(value: Any) -> Any:
    if isinstance(value, _LITERAL_SCALARS):
        return value
    if isinstance(value, list):
        return [_copy_literal(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_copy_literal(item) for item in value)
    if isinstance(value, dict):
        return {_copy_literal(key): _copy_literal(item) for key, item in value.items()}
    raise ValueError("value is outside the static literal whitelist")


def _literal_value(node: ast.AST, environment: dict[str, Any], seen: set[str]) -> Any:
    if isinstance(node, ast.Name):
        return _bound_value(node.id, environment, seen)
    if isinstance(node, ast.Constant) and isinstance(node.value, _LITERAL_SCALARS):
        return node.value
    if isinstance(node, ast.List):
        return [_literal_value(item, environment, seen) for item in node.elts]
    if isinstance(node, ast.Tuple):
        return tuple(_literal_value(item, environment, seen) for item in node.elts)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                nested = _literal_value(value, environment, seen)
                if not isinstance(nested, dict):
                    raise ValueError("dictionary splat is not a dictionary")
                result.update(nested)
            else:
                result[_literal_value(key, environment, seen)] = _literal_value(
                    value, environment, seen
                )
        return result
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, (ast.UAdd, ast.USub))
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
        and not isinstance(node.operand.value, bool)
    ):
        return +node.operand.value if isinstance(node.op, ast.UAdd) else -node.operand.value
    raise ValueError("expression is outside the static literal whitelist")


def literal_value(node: ast.AST, environment: dict[str, Any]) -> Any:
    """Evaluate only explicitly whitelisted literal AST forms and static names."""

    return _literal_value(node, environment, set())


def keyword_nodes(call: ast.Call, environment: dict[str, Any]) -> dict[str, ast.AST] | None:
    """Return statically named keyword syntax, including literal dictionary splats."""

    result: dict[str, ast.AST] = {}
    for keyword in call.keywords:
        if keyword.arg is not None:
            if keyword.arg in result:
                return None
            result[keyword.arg] = keyword.value
            continue
        mapping = resolve_expression(keyword.value, environment)
        if not isinstance(mapping, ast.Dict):
            return None
        for key, value in zip(mapping.keys, mapping.values):
            if key is None:
                return None
            try:
                name = literal_value(key, environment)
            except ValueError:
                return None
            if not isinstance(name, str) or name in result:
                return None
            result[name] = value
    return result
