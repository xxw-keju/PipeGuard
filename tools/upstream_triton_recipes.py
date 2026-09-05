from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Protocol

from upstream_common import RECIPE_VERSION, canonical_json, stable_id
from upstream_pytest_params import is_pytest_raises_block, pytest_parameter_environments
from upstream_static_eval import AMBIGUOUS, bind_once
from upstream_triton_compile import analyze_triton_compile
from upstream_triton_launch import analyze_triton_launch


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


@dataclass(frozen=True)
class TritonRecipeAnalysis:
    recipes: dict[str, list[dict[str, Any]]]
    attempts: tuple[dict[str, Any], ...]


def _mutation_root(node: ast.AST) -> str | None:
    while isinstance(node, (ast.Attribute, ast.Subscript)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class _MutationVisitor(ast.NodeVisitor):
    def __init__(self, *, include_calls: bool) -> None:
        self.names: set[str] = set()
        self.include_calls = include_calls

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.names.add(node.name)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)

    def visit_Lambda(self, node: ast.Lambda) -> None:
        return

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = _mutation_root(node)
            if root is not None:
                self.names.add(root)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if isinstance(node.ctx, (ast.Store, ast.Del)):
            root = _mutation_root(node)
            if root is not None:
                self.names.add(root)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if not self.include_calls:
            return
        if isinstance(node.func, ast.Attribute):
            root = _mutation_root(node.func)
            if root is not None:
                self.names.add(root)
        for value in (*node.args, *(keyword.value for keyword in node.keywords)):
            self.names.update(
                child.id
                for child in ast.walk(value)
                if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
            )
        self.generic_visit(node.func)


def _bound_names(node: ast.AST, *, include_calls: bool = True) -> set[str]:
    if isinstance(node, (ast.Import, ast.ImportFrom)):
        return {alias.asname or alias.name.split(".")[0] for alias in node.names}
    visitor = _MutationVisitor(include_calls=include_calls)
    visitor.visit(node)
    return visitor.names


def _invalidate_bindings(environment: dict[str, Any], names: set[str]) -> None:
    pending = list(names)
    invalidated: set[str] = set()
    while pending:
        name = pending.pop()
        if name in invalidated:
            continue
        invalidated.add(name)
        value = environment.get(name)
        if isinstance(value, ast.Name):
            pending.append(value.id)
        pending.extend(
            bound_name
            for bound_name, bound_value in environment.items()
            if isinstance(bound_value, ast.Name) and bound_value.id == name
        )
    for name in invalidated:
        if name in environment:
            environment[name] = AMBIGUOUS


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
        "origin": (
            f"{getattr(evidence, 'origin_kind', 'static-inference')}:"
            f"{relative_path}:{call.lineno}-{end_line}"
        ),
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


def promote_inferred_root_roles(
    roles: dict[str, str],
    inferred_recipes: dict[str, list[dict[str, Any]]],
) -> dict[str, str]:
    promoted = dict(roles)
    for candidate_id, recipes in inferred_recipes.items():
        if not recipes:
            continue
        role = promoted[candidate_id]
        promoted[candidate_id] = "ROOT_HELPER" if role in {"HELPER", "ROOT_HELPER"} else "ROOT"
    return promoted


def analyze_triton_recipes(
    candidates: list[Any],
    tree: ast.Module,
    aliases: AliasResolver,
    relative_path: str,
) -> TritonRecipeAnalysis:
    """Scan static environments and retain accepted and rejected call evidence."""

    candidates_by_node = {id(candidate.node): candidate for candidate in candidates}
    recipes: dict[str, list[dict[str, Any]]] = defaultdict(list)
    attempts: list[dict[str, Any]] = []

    def inspect_call(call: ast.Call, environment: dict[str, Any]) -> None:
        decision = analyze_triton_compile(call, environment, aliases, candidates_by_node)
        if decision.status == "NOT_APPLICABLE":
            decision = analyze_triton_launch(call, environment, aliases, candidates_by_node)
        if decision.status == "NOT_APPLICABLE":
            return
        attempts.append({
            "line": call.lineno,
            "end_line": call.end_lineno or call.lineno,
            "mechanism": decision.mechanism,
            "status": decision.status,
            "code": decision.code,
            "candidate_id": decision.candidate_id,
        })
        for evidence in (
            (decision.evidence, *decision.additional_evidence)
            if decision.evidence is not None else ()
        ):
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
                _invalidate_bindings(environment, _bound_names(statement))
                continue
            value: ast.AST | None = None
            target: ast.AST | None = None
            if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
                target, value = statement.targets[0], statement.value
            elif isinstance(statement, ast.AnnAssign):
                target, value = statement.target, statement.value
            call: ast.Call | None = None
            if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call):
                call = statement.value
            elif value is not None and isinstance(value, ast.Call):
                call = value
            if call is not None:
                inspect_call(call, environment)
            if isinstance(target, ast.Name) and value is not None:
                bind_once(environment, target.id, value)
            else:
                _invalidate_bindings(
                    environment,
                    _bound_names(statement, include_calls=call is None),
                )
            if call is not None and aliases.qualified(call.func) not in {
                "triton.compile",
                "triton.compiler.ASTSource",
                "triton.compiler.compiler.ASTSource",
                "triton.backends.compiler.GPUTarget",
            }:
                _invalidate_bindings(environment, _bound_names(call))

    scan_block(tree.body, {}, prebind_definitions=True)
    result: dict[str, list[dict[str, Any]]] = {}
    for candidate_id in sorted(recipes):
        unique = {recipe["recipe_id"]: recipe for recipe in recipes[candidate_id]}
        result[candidate_id] = [unique[key] for key in sorted(unique)]
    return TritonRecipeAnalysis(result, tuple(attempts))


def infer_triton_recipes(
    candidates: list[Any],
    tree: ast.Module,
    aliases: AliasResolver,
    relative_path: str,
) -> dict[str, list[dict[str, Any]]]:
    return analyze_triton_recipes(candidates, tree, aliases, relative_path).recipes


def classify_triton_candidate(
    candidate: Any,
    role: str,
    lowerable_recipes: int,
    aliases: AliasResolver,
) -> str:
    if lowerable_recipes:
        return "WITH_RECIPE"
    if role in {"HELPER", "ROOT_HELPER"}:
        return "HELPER"
    if role == "UNCLASSIFIED":
        return "UNCLASSIFIED"
    if candidate.kind == "AUTOTUNE_JIT":
        return "AUTOTUNE_ROOT"
    arguments = candidate.node.args
    parameters = (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)
    if not parameters and arguments.vararg is None and arguments.kwarg is None:
        return "ZERO_ARGUMENT_ROOT"
    constexpr_names = {
        "constexpr",
        "tl.constexpr",
        "triton.language.constexpr",
    }
    if parameters and all(
        aliases.qualified(parameter.annotation) == "triton.language.constexpr"
        or (
            isinstance(parameter.annotation, ast.Constant)
            and isinstance(parameter.annotation.value, str)
            and parameter.annotation.value in constexpr_names
        )
        for parameter in parameters
    ) and arguments.vararg is None and arguments.kwarg is None:
        return "ALL_CONSTEXPR_ROOT"
    return "RUNTIME_PARAMETER_ROOT"
