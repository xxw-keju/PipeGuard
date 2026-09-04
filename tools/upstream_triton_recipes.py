from __future__ import annotations

import ast
from collections import defaultdict
from typing import Any, Protocol

from upstream_common import RECIPE_VERSION, canonical_json, stable_id
from upstream_pytest_params import is_pytest_raises_block, pytest_parameter_environments
from upstream_static_eval import AMBIGUOUS, bind_once
from upstream_triton_compile import triton_compile_evidence


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


def _bound_names(node: ast.AST) -> set[str]:
    names: set[str] = set()
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        names.add(node.name)
    elif isinstance(node, (ast.Import, ast.ImportFrom)):
        for alias in node.names:
            names.add(alias.asname or alias.name.split(".")[0])
    elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
        targets: list[ast.AST] = []
        if isinstance(node, ast.Assign):
            targets.extend(node.targets)
        else:
            targets.append(node.target)
        for target in targets:
            for child in ast.walk(target):
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Store):
                    names.add(child.id)
    return names


def _make_recipe(evidence: Any, call: ast.Call, relative_path: str) -> dict[str, Any]:
    candidate = evidence.candidate
    compiler = evidence.compiler
    end_line = call.end_lineno or call.lineno
    digest = stable_id(
        candidate.candidate_id,
        relative_path,
        call.lineno,
        end_line,
        canonical_json(compiler),
    )[:8]
    return {
        "schema_version": RECIPE_VERSION,
        "recipe_id": f"{candidate.candidate_id}-sm90-L{call.lineno}-{digest}",
        "candidate_id": candidate.candidate_id,
        "dsl": "triton",
        "origin": f"static-inference:{relative_path}:{call.lineno}-{end_line}",
        "lowerable": True,
        "unresolved": [],
        "bindings": {},
        "materialization": {
            "mode": "direct",
            "resolver_kwargs": {},
            "args": [],
            "kwargs": {},
        },
        "compiler": compiler,
        "reference": {"mode": "reference_replay"},
        "target": {"backend": "cuda", "arch": "sm90", "warp_size": 32},
    }


def infer_triton_recipes(
    candidates: list[Any],
    tree: ast.Module,
    aliases: AliasResolver,
    relative_path: str,
) -> dict[str, list[dict[str, Any]]]:
    """Scan conservative static environments for literal SM90 compile recipes."""

    candidates_by_node = {id(candidate.node): candidate for candidate in candidates}
    recipes: dict[str, list[dict[str, Any]]] = defaultdict(list)

    def inspect_call(call: ast.Call, environment: dict[str, Any]) -> None:
        evidence = triton_compile_evidence(call, environment, aliases, candidates_by_node)
        if evidence is not None:
            recipes[evidence.candidate.candidate_id].append(
                _make_recipe(evidence, call, relative_path)
            )

    def scan_block(
        statements: list[ast.stmt],
        inherited: dict[str, Any],
        *,
        prebind_definitions: bool = False,
    ) -> None:
        environment = dict(inherited)
        if prebind_definitions:
            for statement in statements:
                if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    bind_once(environment, statement.name, statement)
        for statement in statements:
            if isinstance(statement, (ast.FunctionDef, ast.AsyncFunctionDef)):
                bind_once(environment, statement.name, statement)
                arguments = statement.args
                base_environment = dict(environment)
                for argument in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs):
                    base_environment[argument.arg] = AMBIGUOUS
                if arguments.vararg:
                    base_environment[arguments.vararg.arg] = AMBIGUOUS
                if arguments.kwarg:
                    base_environment[arguments.kwarg.arg] = AMBIGUOUS
                for parameters in pytest_parameter_environments(statement, aliases):
                    child_environment = dict(base_environment)
                    child_environment.update(parameters)
                    scan_block(statement.body, child_environment)
                continue
            if isinstance(statement, ast.ClassDef):
                bind_once(environment, statement.name, statement)
                scan_block(statement.body, environment)
                continue
            if is_pytest_raises_block(statement, aliases):
                continue
            if isinstance(statement, (ast.With, ast.AsyncWith)):
                continue
            value: ast.AST | None = None
            target: ast.AST | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            elif isinstance(statement, ast.AnnAssign):
                target, value = statement.target, statement.value
            if isinstance(target, ast.Name) and value is not None:
                bind_once(environment, target.id, value)
            elif target is not None:
                for name in _bound_names(statement):
                    environment[name] = AMBIGUOUS
            call: ast.Call | None = None
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
            elif value is not None and isinstance(value, ast.Call):
                call = value
            if call is not None:
                inspect_call(call, environment)

    scan_block(tree.body, {}, prebind_definitions=True)
    result: dict[str, list[dict[str, Any]]] = {}
    for candidate_id in sorted(recipes):
        unique = {recipe["recipe_id"]: recipe for recipe in recipes[candidate_id]}
        result[candidate_id] = [unique[key] for key in sorted(unique)]
    return result
