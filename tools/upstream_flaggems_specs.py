#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import itertools
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from upstream_common import atomic_write, pretty_json, stable_id


SCHEMA_VERSION = "pipeguard-flaggems-specification-audit-v1"
_UNKNOWN = object()


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_value(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise ValueError("value is not JSON-stable")


def _qualified(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified(node.value)
        return f"{parent}.{node.attr}" if parent else None
    if isinstance(node, ast.Call):
        return _qualified(node.func)
    return None


def _eval(node: ast.AST, env: dict[str, Any]) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.Name):
        if node.id not in env:
            raise ValueError(node.id)
        return env[node.id]
    if isinstance(node, ast.Attribute):
        base = _eval(node.value, env)
        if isinstance(base, dict) and node.attr in base:
            return base[node.attr]
        if isinstance(base, str):
            return f"{base}.{node.attr}"
        raise ValueError(node.attr)
    if isinstance(node, (ast.List, ast.Tuple, ast.Set)):
        values = [_eval(item, env) for item in node.elts]
        return tuple(values) if isinstance(node, ast.Tuple) else values
    if isinstance(node, ast.Dict):
        if any(key is None for key in node.keys):
            raise ValueError("dict splat")
        return {_eval(key, env): _eval(value, env) for key, value in zip(node.keys, node.values)}
    if isinstance(node, ast.UnaryOp):
        value = _eval(node.operand, env)
        if isinstance(node.op, ast.USub):
            return -value
        if isinstance(node.op, ast.UAdd):
            return +value
        if isinstance(node.op, ast.Not):
            return not value
        raise ValueError("unary")
    if isinstance(node, ast.BinOp):
        left = _eval(node.left, env)
        right = _eval(node.right, env)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Mod):
            return left % right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError("binary")
    if isinstance(node, ast.Subscript):
        return _eval(node.value, env)[_eval(node.slice, env)]
    if isinstance(node, ast.IfExp):
        return _eval(node.body if _eval(node.test, env) else node.orelse, env)
    if isinstance(node, ast.Compare) and len(node.ops) == len(node.comparators) == 1:
        left = _eval(node.left, env)
        right = _eval(node.comparators[0], env)
        op = node.ops[0]
        if isinstance(op, ast.Eq): return left == right
        if isinstance(op, ast.NotEq): return left != right
        if isinstance(op, ast.Lt): return left < right
        if isinstance(op, ast.LtE): return left <= right
        if isinstance(op, ast.Gt): return left > right
        if isinstance(op, ast.GtE): return left >= right
        if isinstance(op, ast.In): return left in right
        if isinstance(op, ast.NotIn): return left not in right
        raise ValueError("compare")
    if isinstance(node, ast.Call):
        name = _qualified(node.func)
        args = [_eval(item, env) for item in node.args]
        if name in {"range", "builtins.range"}:
            return list(range(*args))
        if name in {"list", "tuple", "set"} and len(args) == 1:
            return list(args[0]) if name != "tuple" else tuple(args[0])
        if name in {"min", "max", "abs", "int", "float", "bool", "len"}:
            return getattr(__builtins__, name)(*args) if not isinstance(__builtins__, dict) else __builtins__[name](*args)
        if name == "getattr" and len(args) >= 2 and isinstance(args[0], str) and isinstance(args[1], str):
            return f"{args[0]}.{args[1]}"
        if name and name.endswith("pytest.param"):
            return tuple(args)
        raise ValueError(name or "call")
    if isinstance(node, ast.ListComp) and len(node.generators) == 1:
        generator = node.generators[0]
        if generator.ifs or generator.is_async or not isinstance(generator.target, ast.Name):
            raise ValueError("list comprehension")
        values = _eval(generator.iter, env)
        result = []
        for value in values:
            child = dict(env)
            child[generator.target.id] = value
            result.append(_eval(node.elt, child))
        return result
    raise ValueError(type(node).__name__)


def _assign(target: ast.AST, value: Any, env: dict[str, Any]) -> bool:
    if isinstance(target, ast.Name):
        env[target.id] = value
        return True
    if isinstance(target, (ast.Tuple, ast.List)) and isinstance(value, (tuple, list)) and len(target.elts) == len(value):
        return all(_assign(item, selected, env) for item, selected in zip(target.elts, value))
    return False


def _module_environment(tree: ast.Module, initial: dict[str, Any]) -> dict[str, Any]:
    env = dict(initial)

    def scan(statements: list[ast.stmt]) -> None:
        for statement in statements:
            if isinstance(statement, ast.Import):
                for alias in statement.names:
                    env[alias.asname or alias.name.split(".")[0]] = alias.name
            elif isinstance(statement, ast.ImportFrom):
                for alias in statement.names:
                    local = alias.asname or alias.name
                    if alias.name == "*":
                        continue
                    env.setdefault(local, f"{statement.module}.{alias.name}")
            elif isinstance(statement, (ast.Assign, ast.AnnAssign)):
                targets = statement.targets if isinstance(statement, ast.Assign) else [statement.target]
                value_node = statement.value
                if value_node is None:
                    continue
                try:
                    value = _eval(value_node, env)
                except Exception:
                    continue
                for target in targets:
                    _assign(target, value, env)
            elif isinstance(statement, ast.If):
                name = _qualified(statement.test)
                if name and name.endswith("QUICK_MODE"):
                    scan(statement.orelse)
                    continue
                try:
                    selected = statement.body if _eval(statement.test, env) else statement.orelse
                except Exception:
                    continue
                scan(selected)
    scan(tree.body)
    return env


def _parameter_names(node: ast.AST, env: dict[str, Any]) -> tuple[str, ...] | None:
    try:
        value = _eval(node, env)
    except Exception:
        return None
    if isinstance(value, str):
        names = tuple(item.strip() for item in value.split(",") if item.strip())
    elif isinstance(value, (list, tuple)) and all(isinstance(item, str) for item in value):
        names = tuple(value)
    else:
        return None
    return names or None


def _parameter_rows(node: ast.AST, names: tuple[str, ...], env: dict[str, Any]) -> list[dict[str, Any]] | None:
    try:
        values = _eval(node, env)
    except Exception:
        return None
    if not isinstance(values, (list, tuple)):
        return None
    rows = []
    for value in values:
        if len(names) == 1:
            selected = (value,)
        elif isinstance(value, (list, tuple)) and len(value) >= len(names):
            selected = tuple(value[:len(names)])
        else:
            return None
        try:
            rows.append({name: _json_value(item) for name, item in zip(names, selected)})
        except ValueError:
            return None
    return rows


def _representative_environments(groups: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    if not groups:
        return [{}]
    choices: list[tuple[int, ...]] = [tuple(0 for _ in groups)]
    for group_index, rows in enumerate(groups):
        for row_index in sorted({1 if len(rows) > 1 else 0, len(rows) // 2, len(rows) - 1}):
            selected = [0 for _ in groups]
            selected[group_index] = row_index
            choices.append(tuple(selected))
    choices.append(tuple(len(rows) - 1 for rows in groups))
    environments: list[dict[str, Any]] = []
    seen: set[str] = set()
    for selected in choices:
        merged: dict[str, Any] = {}
        for group, index in zip(groups, selected):
            merged.update(group[index])
        key = json.dumps(merged, sort_keys=True, separators=(",", ":"))
        if key not in seen:
            seen.add(key)
            environments.append(merged)
    return environments


def _called_qualified_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {
        name
        for node in ast.walk(function)
        if isinstance(node, ast.Call)
        if (name := _qualified(node.func)) is not None
    }


def _called_names(function: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    return {name.split(".")[-1] for name in _called_qualified_names(function)}


class _UnconditionalCallVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self.names: set[str] = set()

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        return

    visit_AsyncFunctionDef = visit_FunctionDef
    visit_ClassDef = visit_FunctionDef
    visit_Lambda = visit_FunctionDef

    def visit_If(self, node: ast.If) -> None:
        return

    visit_For = visit_If
    visit_AsyncFor = visit_If
    visit_While = visit_If
    visit_Try = visit_If
    visit_Match = visit_If

    def visit_Call(self, node: ast.Call) -> None:
        name = _qualified(node.func)
        if name:
            self.names.add(name.split(".")[-1])
        self.generic_visit(node)


def _unconditional_called_names(
    function: ast.FunctionDef | ast.AsyncFunctionDef,
) -> set[str]:
    visitor = _UnconditionalCallVisitor()
    for statement in function.body:
        visitor.visit(statement)
    return visitor.names


def _test_cases(source_root: Path) -> list[dict[str, Any]]:
    accuracy_path = source_root / "tests" / "accuracy_utils.py"
    utility_env: dict[str, Any] = {}
    if accuracy_path.is_file():
        utility_tree = ast.parse(accuracy_path.read_text(), filename=str(accuracy_path))
        utility_env = _module_environment(utility_tree, {"torch": "torch"})
    initial = {
        "torch": "torch",
        "triton": "triton",
        "QUICK_MODE": False,
        "utils": utility_env,
        "cfg": {"QUICK_MODE": False},
    }
    process = subprocess.run(
        ["git", "-C", str(source_root), "ls-files", "tests/*.py", "tests/**/*.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    cases: list[dict[str, Any]] = []
    for relative_path in sorted(set(process.stdout.splitlines())):
        path = source_root / relative_path
        try:
            source = path.read_text(encoding="utf-8")
            tree = ast.parse(source, filename=relative_path)
        except (UnicodeDecodeError, SyntaxError):
            continue
        env = _module_environment(tree, initial)
        lines = source.splitlines(keepends=True)
        for function in [
            node for node in tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name.startswith("test")
        ]:
            decorator_names = {_qualified(item) for item in function.decorator_list}
            calls = _called_qualified_names(function)
            if (
                "pytest.raises" in calls
                or any(name and name.endswith((".xfail", ".skip")) for name in decorator_names)
            ):
                continue
            groups: list[list[dict[str, Any]]] = []
            dimensions: list[str] = []
            complete = True
            for decorator in function.decorator_list:
                if not isinstance(decorator, ast.Call):
                    continue
                name = _qualified(decorator.func)
                if not name or not name.endswith("parametrize") or len(decorator.args) < 2:
                    continue
                names = _parameter_names(decorator.args[0], env)
                rows = _parameter_rows(decorator.args[1], names or (), env) if names else None
                if not names or not rows:
                    complete = False
                    break
                dimensions.extend(names)
                groups.append(rows)
            if not complete:
                continue
            start = min([function.lineno, *(item.lineno for item in function.decorator_list)])
            end = function.end_lineno or function.lineno
            source_sha = hashlib.sha256("".join(lines[start - 1:end]).encode()).hexdigest()
            for index, bindings in enumerate(_representative_environments(groups)):
                cases.append({
                    "test_case_id": f"test-{stable_id(relative_path, function.name, index, bindings)}",
                    "path": relative_path,
                    "function": function.name,
                    "start_line": start,
                    "end_line": end,
                    "source_sha256": source_sha,
                    "called_names": sorted(_called_names(function)),
                    "called_qualified_names": sorted(_called_qualified_names(function)),
                    "parameter_dimensions": sorted(set(dimensions)),
                    "bindings": bindings,
                    "representative_index": index,
                })
    return cases


def _source_entry_names(source_root: Path, relative_path: str, hosts: set[str]) -> set[str]:
    tree = ast.parse((source_root / relative_path).read_text(encoding="utf-8"), filename=relative_path)
    functions: dict[str, ast.FunctionDef | ast.AsyncFunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.setdefault(node.name, node)
    reverse: dict[str, set[str]] = defaultdict(set)
    names = set(functions)
    for owner, function in functions.items():
        for called in _unconditional_called_names(function) & names:
            reverse[called].add(owner)
    result = set(hosts)
    pending = list(hosts)
    while pending:
        callee = pending.pop()
        for caller in reverse.get(callee, ()):
            if caller not in result:
                result.add(caller)
                pending.append(caller)
    result.add(Path(relative_path).stem)
    return result


def _case_matches(case: dict[str, Any], *, source_stem: str, entry_names: set[str]) -> bool:
    test_stem = Path(case["path"]).stem
    normalized = source_stem.lstrip("_")
    same_file_family = (
        test_stem in {f"test_{normalized}", f"test_{source_stem}"}
        or test_stem.startswith(f"test_{normalized}_")
    )
    named_test = normalized in case["function"]
    direct_call = bool(set(case["called_names"]) & entry_names)
    # Names such as randn/max/forward occur pervasively as helpers in unrelated
    # tests.  Require operation-local provenance before accepting the call-name
    # match as specification evidence.
    return direct_call and (same_file_family or named_test)


def _diff_dimensions(original: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    keys = sorted(set(original) | set(candidate))
    return [key for key in keys if original.get(key, _UNKNOWN) != candidate.get(key, _UNKNOWN)]


def _select_from_test_matrix(cases: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    unique: dict[str, dict[str, Any]] = {}
    for case in cases:
        key = json.dumps(case["bindings"], sort_keys=True, separators=(",", ":"))
        unique.setdefault(key, case)
    ordered = sorted(unique.values(), key=lambda item: (
        item["representative_index"], item["test_case_id"]
    ))
    if not ordered:
        return None, [], []
    original = ordered[0]
    remaining = ordered[1:]
    selected: list[dict[str, Any]] = []
    covered: set[str] = set()
    while remaining and len(selected) < 3:
        ranked = sorted(
            remaining,
            key=lambda item: (
                -len(set(_diff_dimensions(original["bindings"], item["bindings"])) - covered),
                -len(_diff_dimensions(original["bindings"], item["bindings"])),
                item["representative_index"], item["test_case_id"],
            ),
        )
        chosen = ranked[0]
        dimensions = _diff_dimensions(original["bindings"], chosen["bindings"])
        if not dimensions:
            remaining.remove(chosen)
            continue
        chosen = {**chosen, "boundary_dimensions": dimensions}
        selected.append(chosen)
        covered.update(dimensions)
        remaining = [item for item in remaining if item["test_case_id"] != chosen["test_case_id"]]
    return original, selected, sorted(covered)


def _select_specs(root: dict[str, Any], cases: list[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    by_test: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        by_test[(case["path"], case["function"])].append(case)
    attempts = [
        _select_from_test_matrix(by_test[key])
        for key in sorted(by_test)
    ]
    complete = [
        attempt for attempt in attempts
        if len(attempt[1]) >= 3 and len(attempt[2]) >= 2
    ]
    if complete:
        return complete[0]
    if not attempts:
        return None, [], []
    return max(
        attempts,
        key=lambda attempt: (len(attempt[1]), len(attempt[2])),
    )


def recover_specifications(
    *,
    source_root: Path,
    root_audit_path: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    root_audit_path = root_audit_path.resolve(strict=True)
    audit = _load(root_audit_path)
    occurrences = {item["occurrence_id"]: item for item in audit["launch_occurrences"]}
    tests = _test_cases(source_root)
    tests_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in tests:
        tests_by_path[case["path"]].append(case)

    roots: list[dict[str, Any]] = []
    specifications: list[dict[str, Any]] = []
    for root in audit["roots"]:
        if root["audit_status"] != "READY_FOR_RECIPE_RECOVERY":
            roots.append({
                "root_id": root["root_id"],
                "candidate_id": root["candidate_id"],
                "status": root["audit_status"],
                "entry_names": [],
                "matched_test_cases": 0,
                "original_spec_id": None,
                "boundary_spec_ids": [],
                "boundary_dimensions": [],
            })
            continue
        root_occurrences = [occurrences[item] for item in root["launch_occurrence_ids"]]
        unconditional_hosts = {
            item["host_qualname"].split(".")[-1]
            for item in root_occurrences
            if not item.get("branch_predicates", [])
        }
        entry_names = _source_entry_names(
            source_root,
            root["path"],
            unconditional_hosts,
        ) if unconditional_hosts else set()
        source_stem = Path(root["path"]).stem
        matched = [
            case for case in tests
            if _case_matches(
                case,
                source_stem=source_stem,
                entry_names=unconditional_hosts,
            )
        ]
        original, boundaries, dimensions = _select_specs(root, matched)
        if original is None:
            status = "NO_OFFICIAL_TEST_RECIPE"
        elif len(boundaries) < 3 or len(dimensions) < 2:
            status = "INSUFFICIENT_BOUNDARY_EVIDENCE"
        else:
            status = "SOURCE_REPLAY_READY"
        original_id = None
        boundary_ids: list[str] = []
        if original is not None:
            original_id = f"{root['root_id']}-original-{stable_id(original['test_case_id'], original['bindings'], length=10)}"
            specifications.append({
                "spec_id": original_id,
                "root_id": root["root_id"],
                "candidate_id": root["candidate_id"],
                "role": "ORIGINAL",
                "parent_spec_id": None,
                "test_case_id": original["test_case_id"],
                "test_path": original["path"],
                "test_function": original["function"],
                "test_start_line": original["start_line"],
                "test_end_line": original["end_line"],
                "test_source_sha256": original["source_sha256"],
                "bindings": original["bindings"],
                "boundary_dimensions": [],
                "launch_occurrence_ids": root["launch_occurrence_ids"],
                "materialization_status": "TOOLCHAIN_UNAVAILABLE",
                "expected_pipeline": "UNASSESSED",
            })
            for boundary in boundaries:
                spec_id = f"{root['root_id']}-boundary-{stable_id(boundary['test_case_id'], boundary['bindings'], length=10)}"
                boundary_ids.append(spec_id)
                specifications.append({
                    "spec_id": spec_id,
                    "root_id": root["root_id"],
                    "candidate_id": root["candidate_id"],
                    "role": "BOUNDARY",
                    "parent_spec_id": original_id,
                    "test_case_id": boundary["test_case_id"],
                    "test_path": boundary["path"],
                    "test_function": boundary["function"],
                    "test_start_line": boundary["start_line"],
                    "test_end_line": boundary["end_line"],
                    "test_source_sha256": boundary["source_sha256"],
                    "bindings": boundary["bindings"],
                    "boundary_dimensions": boundary["boundary_dimensions"],
                    "launch_occurrence_ids": root["launch_occurrence_ids"],
                    "materialization_status": "TOOLCHAIN_UNAVAILABLE",
                    "expected_pipeline": "UNASSESSED",
                })
        roots.append({
            "root_id": root["root_id"],
            "candidate_id": root["candidate_id"],
            "status": status,
            "entry_names": sorted(entry_names),
            "matched_test_cases": len(matched),
            "original_spec_id": original_id,
            "boundary_spec_ids": boundary_ids,
            "boundary_dimensions": dimensions,
        })

    status_counts = Counter(item["status"] for item in roots)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": audit["source_id"],
        "commit": audit["commit"],
        "root_audit": {
            "path": "launch-root-audit.json",
            "sha256": _sha(root_audit_path),
        },
        "counts": {
            "official_test_representatives": len(tests),
            "audited_roots": len(roots),
            "root_status": dict(sorted(status_counts.items())),
            "original_specifications": sum(item["role"] == "ORIGINAL" for item in specifications),
            "boundary_specifications": sum(item["role"] == "BOUNDARY" for item in specifications),
            "source_replay_ready_roots": status_counts["SOURCE_REPLAY_READY"],
            "native_materialized_roots": 0,
            "safe_roots": 0,
        },
        "roots": sorted(roots, key=lambda item: item["root_id"]),
        "specifications": sorted(specifications, key=lambda item: item["spec_id"]),
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Recover source-grounded FlagGems specifications")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--root-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    result = recover_specifications(
        source_root=arguments.source_root,
        root_audit_path=arguments.root_audit,
    )
    atomic_write(arguments.output, pretty_json(result))
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
