from __future__ import annotations

import ast
import math
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
                literal_key = _literal_value(key, environment, seen)
                try:
                    result[literal_key] = _literal_value(value, environment, seen)
                except TypeError as error:
                    raise ValueError("literal dictionary key is not hashable") from error
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


def _json_stable(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("non-finite float is not stable JSON")
    if isinstance(value, _LITERAL_SCALARS):
        return value
    if isinstance(value, list):
        return [_json_stable(item) for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("JSON object keys must be strings")
        return {key: _json_stable(item) for key, item in value.items()}
    raise ValueError("value cannot round-trip through JSON without type loss")


def json_literal_value(node: ast.AST, environment: dict[str, Any]) -> Any:
    """Evaluate a literal only when JSON recipe serialization preserves its type."""

    return _json_stable(literal_value(node, environment))


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


def closed_value(
    node: ast.AST,
    environment: dict[str, Any],
    call_evaluator: Any = None,
    _seen: set[str] | None = None,
) -> Any:
    """Evaluate a closed expression with no Python execution capability."""

    seen = set() if _seen is None else _seen
    if isinstance(node, ast.Name):
        if node.id in seen:
            raise ValueError("cyclic closed binding")
        value = environment.get(node.id, _MISSING)
        if value is _MISSING or value is AMBIGUOUS:
            raise ValueError("unresolved closed binding")
        if isinstance(value, ast.AST):
            return closed_value(value, environment, call_evaluator, seen | {node.id})
        return _copy_literal(value)
    if isinstance(node, ast.Constant) and isinstance(node.value, _LITERAL_SCALARS):
        return node.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [closed_value(item, environment, call_evaluator, seen) for item in node.elts]
        return values if isinstance(node, ast.List) else tuple(values)
    if isinstance(node, ast.Dict):
        result: dict[Any, Any] = {}
        for key, value in zip(node.keys, node.values):
            if key is None:
                raise ValueError("closed dictionary splat is unsupported")
            try:
                result[closed_value(key, environment, call_evaluator, seen)] = closed_value(
                    value, environment, call_evaluator, seen
                )
            except TypeError as error:
                raise ValueError("closed dictionary key is not hashable") from error
        return result
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub, ast.Not)):
        value = closed_value(node.operand, environment, call_evaluator, seen)
        if isinstance(node.op, ast.Not):
            return not value
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError("closed unary operand is not numeric")
        return +value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)
    ):
        left = closed_value(node.left, environment, call_evaluator, seen)
        right = closed_value(node.right, environment, call_evaluator, seen)
        if not isinstance(left, int) or isinstance(left, bool) or not isinstance(right, int) or isinstance(right, bool):
            raise ValueError("closed binary operands are not integers")
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and right == 0:
            raise ValueError("closed division by zero")
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        return left % right
    if isinstance(node, ast.BoolOp) and isinstance(node.op, (ast.And, ast.Or)):
        result: Any = None
        for value in node.values:
            result = closed_value(value, environment, call_evaluator, seen)
            if isinstance(node.op, ast.And) and not result:
                return result
            if isinstance(node.op, ast.Or) and result:
                return result
        return result
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = closed_value(node.left, environment, call_evaluator, seen)
        right = closed_value(node.comparators[0], environment, call_evaluator, seen)
        operation = node.ops[0]
        try:
            if isinstance(operation, ast.Eq):
                return left == right
            if isinstance(operation, ast.NotEq):
                return left != right
            if isinstance(operation, ast.Lt):
                return left < right
            if isinstance(operation, ast.LtE):
                return left <= right
            if isinstance(operation, ast.Gt):
                return left > right
            if isinstance(operation, ast.GtE):
                return left >= right
        except TypeError as error:
            raise ValueError("closed comparison operands are incompatible") from error
        raise ValueError("closed comparison is unsupported")
    if isinstance(node, ast.IfExp):
        condition = closed_value(node.test, environment, call_evaluator, seen)
        branch = node.body if condition else node.orelse
        return closed_value(branch, environment, call_evaluator, seen)
    if isinstance(node, ast.Subscript):
        value = closed_value(node.value, environment, call_evaluator, seen)
        index = closed_value(node.slice, environment, call_evaluator, seen)
        if not isinstance(value, (tuple, list, dict)):
            raise ValueError("closed subscript base is unsupported")
        try:
            return _copy_literal(value[index])
        except (KeyError, IndexError, TypeError) as error:
            raise ValueError("closed subscript is unresolved") from error
    if isinstance(node, ast.Call) and call_evaluator is not None:
        if any(isinstance(value, ast.Starred) for value in node.args) or any(
            keyword.arg is None for keyword in node.keywords
        ):
            raise ValueError("closed call splat is unsupported")
        arguments = [closed_value(value, environment, call_evaluator, seen) for value in node.args]
        keywords = {
            keyword.arg: closed_value(keyword.value, environment, call_evaluator, seen)
            for keyword in node.keywords
        }
        return _copy_literal(call_evaluator(node.func, arguments, keywords))
    raise ValueError("expression is outside the closed-expression whitelist")
