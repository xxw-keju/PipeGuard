from __future__ import annotations

import ast
from itertools import product
from typing import Any, Protocol

from upstream_static_eval import literal_value


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


def _has_xfail(node: ast.AST, aliases: AliasResolver) -> bool:
    return any(
        isinstance(child, ast.Call)
        and aliases.qualified(child.func) in {"pytest.mark.xfail", "pytest.param"}
        for child in ast.walk(node)
    )


def _parameter_names(node: ast.AST) -> tuple[str, ...] | None:
    try:
        value = literal_value(node, {})
    except ValueError:
        return None
    if not isinstance(value, str):
        return None
    names = tuple(part.strip() for part in value.split(","))
    if not names or any(not name.isidentifier() for name in names):
        return None
    if len(names) != len(set(names)):
        return None
    return names


def _parameter_rows(node: ast.AST, arity: int, aliases: AliasResolver) -> list[tuple[Any, ...]] | None:
    if _has_xfail(node, aliases):
        return None
    try:
        values = literal_value(node, {})
    except ValueError:
        return None
    if not isinstance(values, (list, tuple)):
        return None
    rows: list[tuple[Any, ...]] = []
    for value in values:
        if arity == 1:
            rows.append((value,))
        elif isinstance(value, (list, tuple)) and len(value) == arity:
            rows.append(tuple(value))
        else:
            return None
    return rows


def pytest_parameter_environments(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
    aliases: AliasResolver,
) -> list[dict[str, Any]]:
    """Expand only literal direct pytest parametrization; unsupported evidence yields none."""

    parameter_sets: list[tuple[tuple[str, ...], list[tuple[Any, ...]]]] = []
    used_names: set[str] = set()
    function_arguments = {
        argument.arg
        for argument in (
            *function.args.posonlyargs,
            *function.args.args,
            *function.args.kwonlyargs,
        )
    }
    for decorator in function.decorator_list:
        qualified = aliases.qualified(decorator)
        if qualified == "pytest.mark.xfail":
            return []
        if qualified != "pytest.mark.parametrize":
            continue
        if not isinstance(decorator, ast.Call) or len(decorator.args) != 2:
            return []
        if any(keyword.arg == "indirect" for keyword in decorator.keywords):
            return []
        if decorator.keywords:
            return []
        names = _parameter_names(decorator.args[0])
        if names is None or not set(names) <= function_arguments or used_names & set(names):
            return []
        rows = _parameter_rows(decorator.args[1], len(names), aliases)
        if rows is None:
            return []
        used_names.update(names)
        parameter_sets.append((names, rows))
    if not parameter_sets:
        return [{}]
    if (
        used_names != function_arguments
        or function.args.vararg is not None
        or function.args.kwarg is not None
    ):
        return []
    environments: list[dict[str, Any]] = []
    for selected_rows in product(*(rows for _names, rows in parameter_sets)):
        environment: dict[str, Any] = {}
        for (names, _rows), row in zip(parameter_sets, selected_rows):
            environment.update(zip(names, row))
        environments.append(environment)
    return environments


def is_pytest_raises_block(statement: ast.stmt, aliases: AliasResolver) -> bool:
    """Identify expected-failure bodies so their compile calls cannot become recipes."""

    if not isinstance(statement, (ast.With, ast.AsyncWith)):
        return False
    return any(
        isinstance(item.context_expr, ast.Call)
        and aliases.qualified(item.context_expr.func) == "pytest.raises"
        for item in statement.items
    )
