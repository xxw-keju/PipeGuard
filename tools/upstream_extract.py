#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import builtins
import json
import sys
import textwrap
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from upstream_common import (
    CAPSULE_MANIFEST_VERSION,
    DSL_LOCKS,
    RECIPE_VERSION,
    SOURCE_INDEX_VERSION,
    TOOL_VERSION,
    atomic_write,
    canonical_json,
    git_blob_hash,
    pretty_json,
    require_locked_checkout,
    safe_component,
    sha256_text,
    stable_id,
)
from upstream_tilelang_recipes import classify_tilelang_recipe, tilelang_parent_map
from upstream_triton_recipes import (
    analyze_triton_recipes,
    infer_triton_recipes,
    classify_triton_candidate,
    promote_inferred_root_roles,
)


DEFAULT_INCLUDE = {
    "tilelang": (
        "examples/",
        "testing/",
        "benchmark/",
        "docs/tutorials/",
    ),
    "triton": (
        "python/examples/",
        "python/test/",
        "python/tutorials/",
        "python/triton_kernels/tests/",
    ),
}


@dataclass(frozen=True)
class Candidate:
    dsl: str
    relative_path: str
    qualname: str
    name: str
    kind: str
    node: ast.FunctionDef | ast.AsyncFunctionDef
    lexical_parents: tuple[str, ...]
    decorator_names: tuple[str, ...]
    start_line: int
    end_line: int
    source_text: str
    candidate_id: str
    free_names: tuple[str, ...]


class AliasTable:
    def __init__(self, tree: ast.Module):
        self.names: dict[str, str] = {}
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    bound = alias.asname or alias.name.split(".")[0]
                    self.names[bound] = alias.name if alias.asname else bound
            elif isinstance(node, ast.ImportFrom) and node.module:
                for alias in node.names:
                    if alias.name == "*":
                        continue
                    self.names[alias.asname or alias.name] = f"{node.module}.{alias.name}"

    def qualified(self, node: ast.AST) -> str | None:
        if isinstance(node, ast.Call):
            return self.qualified(node.func)
        if isinstance(node, ast.Name):
            return self.names.get(node.id, node.id)
        if isinstance(node, ast.Attribute):
            base = self.qualified(node.value)
            return f"{base}.{node.attr}" if base else None
        return None


class CandidateVisitor(ast.NodeVisitor):
    def __init__(self, dsl: str, relative_path: str, source: str, aliases: AliasTable):
        self.dsl = dsl
        self.relative_path = relative_path
        self.source = source
        self.lines = source.splitlines(keepends=True)
        self.aliases = aliases
        self.scope: list[str] = []
        self.candidates: list[Candidate] = []

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._visit_function(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._visit_function(node)

    def _visit_function(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        decorator_names = tuple(
            name for decorator in node.decorator_list
            if (name := self.aliases.qualified(decorator)) is not None
        )
        kind = classify_candidate(self.dsl, decorator_names)
        qualname = ".".join((*self.scope, node.name))
        if kind is not None:
            start_line = min(
                [node.lineno, *(decorator.lineno for decorator in node.decorator_list)]
            )
            end_line = node.end_lineno or node.lineno
            source_text = "".join(self.lines[start_line - 1:end_line])
            free_names = tuple(sorted(function_free_names(node)))
            candidate_id = stable_id(
                self.dsl,
                DSL_LOCKS[self.dsl]["commit"],
                self.relative_path,
                qualname,
                kind,
                start_line,
                end_line,
                sha256_text(source_text),
            )
            self.candidates.append(Candidate(
                dsl=self.dsl,
                relative_path=self.relative_path,
                qualname=qualname,
                name=node.name,
                kind=kind,
                node=node,
                lexical_parents=tuple(self.scope),
                decorator_names=decorator_names,
                start_line=start_line,
                end_line=end_line,
                source_text=source_text,
                candidate_id=candidate_id,
                free_names=free_names,
            ))
        self.scope.append(node.name)
        self.generic_visit(node)
        self.scope.pop()


class NameUseVisitor(ast.NodeVisitor):
    def __init__(self):
        self.loaded: set[str] = set()
        self.bound: set[str] = set()

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load):
            self.loaded.add(node.id)
        elif isinstance(node.ctx, (ast.Store, ast.Param)):
            self.bound.add(node.id)

    def visit_arg(self, node: ast.arg) -> None:
        self.bound.add(node.arg)
        if node.annotation:
            self.visit(node.annotation)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.bound.add(node.name)
        for decorator in node.decorator_list:
            self.visit(decorator)
        for default in (*node.args.defaults, *node.args.kw_defaults):
            if default:
                self.visit(default)
        for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
            self.visit(arg)
        if node.args.vararg:
            self.visit(node.args.vararg)
        if node.args.kwarg:
            self.visit(node.args.kwarg)
        if node.returns:
            self.visit(node.returns)
        for statement in node.body:
            self.visit(statement)

    visit_AsyncFunctionDef = visit_FunctionDef


def function_free_names(node: ast.FunctionDef | ast.AsyncFunctionDef) -> set[str]:
    visitor = NameUseVisitor()
    for decorator in node.decorator_list:
        visitor.visit(decorator)
    for default in (*node.args.defaults, *node.args.kw_defaults):
        if default:
            visitor.visit(default)
    for arg in (*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs):
        visitor.visit(arg)
    if node.args.vararg:
        visitor.visit(node.args.vararg)
    if node.args.kwarg:
        visitor.visit(node.args.kwarg)
    if node.returns:
        visitor.visit(node.returns)
    for statement in node.body:
        visitor.visit(statement)
    return visitor.loaded - visitor.bound - set(dir(builtins))


def classify_candidate(dsl: str, decorators: tuple[str, ...]) -> str | None:
    lowered = tuple(name.lower() for name in decorators)
    if dsl == "tilelang":
        has_primfunc = any(
            name == "tilelang.language.prim_func"
            or name.startswith("tilelang.language.") and name.endswith(".prim_func")
            or name == "tvm.tl.language.prim_func"
            or name.endswith(".tirx.prim_func")
            or name in {"t.prim_func", "tirx.prim_func"}
            for name in lowered
        )
        has_jit = any(name == "tilelang.jit" or name.endswith(".tilelang.jit") for name in lowered)
        has_autotune = any(name == "tilelang.autotune" or name.endswith(".autotune") for name in lowered)
        if has_autotune and has_jit:
            return "AUTOTUNE_JIT"
        if has_jit:
            return "JIT_ENTRY"
        if has_primfunc:
            return "PRIMFUNC"
        return None
    has_jit = any(name == "triton.jit" or name.endswith(".triton.jit") for name in lowered)
    if not has_jit:
        return None
    if any(name == "triton.autotune" or name.endswith(".autotune") for name in lowered):
        return "AUTOTUNE_JIT"
    if any(name == "triton.heuristics" or name.endswith(".heuristics") for name in lowered):
        return "HEURISTICS_JIT"
    return "JIT_ENTRY"


def tracked_python_files(root: Path, dsl: str) -> list[Path]:
    import subprocess

    process = subprocess.run(
        ["git", "-C", str(root), "ls-files", "*.py"],
        check=True,
        capture_output=True,
        text=True,
    )
    prefixes = DEFAULT_INCLUDE[dsl]
    return [
        root / relative
        for relative in sorted(process.stdout.splitlines())
        if relative.startswith(prefixes)
    ]


def bound_names(node: ast.AST) -> set[str]:
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


def node_loaded_names(node: ast.AST) -> set[str]:
    loaded = {
        child.id for child in ast.walk(node)
        if isinstance(child, ast.Name) and isinstance(child.ctx, ast.Load)
    }
    return loaded - set(dir(builtins))


def module_providers(tree: ast.Module) -> dict[str, ast.stmt]:
    providers: dict[str, ast.stmt] = {}
    for statement in tree.body:
        for name in bound_names(statement):
            providers[name] = statement
    return providers


def dependency_nodes(candidate: Candidate, tree: ast.Module) -> tuple[list[ast.stmt], set[str]]:
    providers = module_providers(tree)
    selected: dict[int, ast.stmt] = {}
    unresolved: set[str] = set()
    queue = deque(candidate.free_names)
    seen: set[str] = set()
    while queue:
        name = queue.popleft()
        if name in seen:
            continue
        seen.add(name)
        provider = providers.get(name)
        if provider is None:
            unresolved.add(name)
            continue
        if provider is candidate.node:
            continue
        key = id(provider)
        if key not in selected:
            selected[key] = provider
            queue.extend(node_loaded_names(provider) - bound_names(provider))
    for statement in tree.body:
        if isinstance(statement, ast.ImportFrom) and statement.module == "__future__":
            selected[id(statement)] = statement
    return sorted(selected.values(), key=lambda node: (node.lineno, node.end_lineno or node.lineno)), unresolved


def source_for_node(source_lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", 1)
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        decorators = getattr(node, "decorator_list", ())
        if decorators:
            start = min(start, *(decorator.lineno for decorator in decorators))
    end = getattr(node, "end_lineno", start)
    return "".join(source_lines[start - 1:end])


def candidate_roles(candidates: list[Candidate], tree: ast.Module) -> tuple[dict[str, str], dict[str, list[str]]]:
    by_name: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        by_name[candidate.name].append(candidate)
    root_ids: set[str] = set()
    helper_ids: set[str] = set()
    dependencies: dict[str, set[str]] = defaultdict(set)
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        target = node.func
        if isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name):
            for candidate in by_name.get(target.value.id, ()):
                root_ids.add(candidate.candidate_id)
        elif isinstance(target, ast.Attribute) and target.attr in {"compile", "warmup"}:
            for argument in node.args:
                if isinstance(argument, ast.Name):
                    for candidate in by_name.get(argument.id, ()):
                        root_ids.add(candidate.candidate_id)
    for owner in candidates:
        loaded = node_loaded_names(owner.node)
        for name in loaded:
            for dependency in by_name.get(name, ()):
                if dependency.candidate_id == owner.candidate_id:
                    continue
                dependencies[owner.candidate_id].add(dependency.candidate_id)
                helper_ids.add(dependency.candidate_id)
    roles: dict[str, str] = {}
    for candidate in candidates:
        is_root = candidate.candidate_id in root_ids
        is_helper = candidate.candidate_id in helper_ids
        roles[candidate.candidate_id] = (
            "ROOT_HELPER" if is_root and is_helper else
            "ROOT" if is_root else
            "HELPER" if is_helper else
            "UNCLASSIFIED"
        )
    return roles, {key: sorted(value) for key, value in dependencies.items()}


def make_capsule(
    candidate: Candidate,
    dependencies: list[ast.stmt],
    unresolved: set[str],
    source_lines: list[str],
) -> str:
    header = [
        "# Generated by PipeGuard upstream_extract.py; do not edit.",
        f"# source: {candidate.relative_path}:{candidate.start_line}-{candidate.end_line}",
        f"# candidate: {candidate.candidate_id} ({candidate.qualname})",
        "",
    ]
    chunks: list[str] = []
    for node in dependencies:
        text = source_for_node(source_lines, node)
        if text.strip():
            chunks.append(text.rstrip() + "\n")
    candidate_text = textwrap.dedent(candidate.source_text).rstrip() + "\n"
    if candidate.lexical_parents:
        assignments = "".join(
            f"    {name} = pipeguard_bindings[{name!r}]\n" for name in sorted(unresolved)
        )
        body = textwrap.indent(candidate_text, "    ")
        chunks.append(
            "def pipeguard_resolve(**pipeguard_bindings):\n"
            + assignments
            + body
            + f"    return {candidate.name}\n"
        )
    else:
        chunks.append(candidate_text)
        chunks.append(f"\ndef pipeguard_resolve(**pipeguard_bindings):\n    return {candidate.name}\n")
    return "\n".join(header) + "\n".join(chunks).rstrip() + "\n"


def extract_dsl(
    dsl: str,
    source_root: Path,
    output_root: Path,
    *,
    allow_commit_mismatch: bool,
    triton_rejection_report: Path | None = None,
) -> dict[str, Any]:
    actual_commit = require_locked_checkout(dsl, source_root, allow_commit_mismatch=allow_commit_mismatch)
    corpus_name = f"{dsl}-{DSL_LOCKS[dsl]['commit'][:8]}"
    corpus_root = output_root / corpus_name
    overlay_path = output_root / "curated-recipes.json"
    overlay_document = json.loads(overlay_path.read_text()) if overlay_path.exists() else {
        "schema_version": "pipeguard-upstream-curated-recipes-v1",
        "recipes": [],
    }
    if set(overlay_document) != {"schema_version", "recipes"} or overlay_document.get("schema_version") != "pipeguard-upstream-curated-recipes-v1":
        raise RuntimeError(f"invalid curated recipe overlay: {overlay_path}")
    overlays_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for overlay in overlay_document["recipes"]:
        if overlay.get("dsl") == dsl:
            overlays_by_candidate[overlay.get("candidate_id", "")].append(overlay)
    all_records: list[dict[str, Any]] = []
    totals = Counter()
    triton_attempts: list[dict[str, Any]] = []
    triton_candidates: list[dict[str, Any]] = []
    for path in tracked_python_files(source_root, dsl):
        relative = path.relative_to(source_root).as_posix()
        source = path.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative)
        aliases = AliasTable(tree)
        visitor = CandidateVisitor(dsl, relative, source, aliases)
        visitor.visit(tree)
        candidates = visitor.candidates
        roles, kernel_dependencies = candidate_roles(candidates, tree)
        triton_analysis = (
            analyze_triton_recipes(candidates, tree, aliases, relative)
            if dsl == "triton" else None
        )
        inferred_triton_recipes = triton_analysis.recipes if triton_analysis is not None else {}
        if triton_analysis is not None:
            triton_attempts.extend({"relative_path": relative, **attempt} for attempt in triton_analysis.attempts)
            roles = promote_inferred_root_roles(roles, inferred_triton_recipes)
        tilelang_parents = tilelang_parent_map(tree) if dsl == "tilelang" else {}
        source_lines = source.splitlines(keepends=True)
        blob_hash = git_blob_hash(source_root, Path(relative))
        totals["source_files"] += 1
        totals["candidates"] += len(candidates)
        for candidate in candidates:
            dependencies, unresolved = dependency_nodes(candidate, tree)
            capsule = make_capsule(candidate, dependencies, unresolved, source_lines)
            source_directory = corpus_root / "capsules" / Path(relative)
            filename = (
                f"L{candidate.start_line}-L{candidate.end_line}__"
                f"{safe_component(candidate.qualname)}__{candidate.candidate_id}.py"
            )
            capsule_path = source_directory / filename
            manifest_path = corpus_root / "manifests" / f"{candidate.candidate_id}.json"
            recipe_directory = corpus_root / "recipes" / candidate.candidate_id
            unresolved_recipe_path = recipe_directory / "unresolved.json"
            atomic_write(capsule_path, capsule)
            capsule_relative = capsule_path.relative_to(output_root).as_posix()
            manifest_relative = manifest_path.relative_to(output_root).as_posix()
            recipe_paths = [unresolved_recipe_path]
            curated_recipes = sorted(
                overlays_by_candidate.get(candidate.candidate_id, []),
                key=lambda item: item["recipe_id"],
            )
            tilelang_classification = (
                classify_tilelang_recipe(
                    candidate,
                    aliases,
                    unresolved,
                    tilelang_parents,
                )
                if dsl == "tilelang"
                else None
            )
            if dsl == "triton" and not unresolved:
                curated_recipes.extend(inferred_triton_recipes.get(candidate.candidate_id, []))
                curated_recipes.sort(key=lambda item: item["recipe_id"])
            if (
                dsl == "tilelang"
                and tilelang_classification is not None
                and tilelang_classification.kind == "POSITIVE"
                and not curated_recipes
            ):
                curated_recipes.append({
                    "schema_version": RECIPE_VERSION,
                    "recipe_id": f"{candidate.candidate_id}-sm90a-static-primfunc",
                    "candidate_id": candidate.candidate_id,
                    "dsl": dsl,
                    "origin": "static-inference:no-unresolved-primfunc-bindings",
                    "lowerable": True,
                    "unresolved": [],
                    "bindings": {},
                    "materialization": {
                        "mode": "direct",
                        "resolver_kwargs": {},
                        "args": [],
                        "kwargs": {},
                    },
                    "compiler": {"target_host": None, "pass_configs": {}},
                    "reference": {"mode": "reference_replay"},
                    "target": {"backend": "cuda", "arch": "sm_90a", "warp_size": 32},
                })
            for curated in curated_recipes:
                if curated.get("candidate_id") != candidate.candidate_id or curated.get("dsl") != dsl:
                    raise RuntimeError(f"curated recipe identity mismatch for {candidate.candidate_id}")
                recipe_paths.append(recipe_directory / f"{safe_component(curated['recipe_id'])}.json")
            recipe_relatives = [path.relative_to(output_root).as_posix() for path in recipe_paths]
            manifest = {
                "schema_version": CAPSULE_MANIFEST_VERSION,
                "candidate_id": candidate.candidate_id,
                "dsl": dsl,
                "upstream": {
                    "commit": actual_commit,
                    "relative_path": relative,
                    "git_blob": blob_hash,
                    "source_sha256": sha256_text(source),
                    "selected_source_sha256": sha256_text(candidate.source_text),
                    "start_line": candidate.start_line,
                    "end_line": candidate.end_line,
                    "qualname": candidate.qualname,
                },
                "candidate_kind": candidate.kind,
                "role": roles[candidate.candidate_id],
                "decorators": list(candidate.decorator_names),
                "lexical_parents": list(candidate.lexical_parents),
                "kernel_dependencies": kernel_dependencies.get(candidate.candidate_id, []),
                "free_names": list(candidate.free_names),
                "unresolved_bindings": sorted(unresolved),
                "capsule": {
                    "path": capsule_relative,
                    "sha256": sha256_text(capsule),
                    "resolver": "pipeguard_resolve",
                    "source_map": {
                        "logical_path": relative,
                        "start_line": candidate.start_line,
                        "end_line": candidate.end_line,
                    },
                },
                "recipe_paths": recipe_relatives,
            }
            recipe = {
                "schema_version": RECIPE_VERSION,
                "recipe_id": f"{candidate.candidate_id}-unresolved",
                "candidate_id": candidate.candidate_id,
                "dsl": dsl,
                "origin": "static-discovery",
                "lowerable": False,
                "unresolved": sorted(set(unresolved) | {"concrete_specialization"}),
                "bindings": {},
                "target": {
                    "backend": "cuda",
                    "arch": "sm_90a" if dsl == "tilelang" else "sm90",
                    "warp_size": 32,
                },
            }
            atomic_write(manifest_path, pretty_json(manifest))
            atomic_write(unresolved_recipe_path, pretty_json(recipe))
            for curated_path, curated in zip(recipe_paths[1:], curated_recipes):
                atomic_write(curated_path, pretty_json(curated))
            lowerable_recipe_count = sum(1 for item in curated_recipes if item.get("lowerable"))
            if dsl == "triton":
                triton_candidates.append({
                    "candidate_id": candidate.candidate_id,
                    "relative_path": relative,
                    "qualname": candidate.qualname,
                    "kind": candidate.kind,
                    "role": roles[candidate.candidate_id],
                    "bucket": classify_triton_candidate(
                        candidate,
                        roles[candidate.candidate_id],
                        lowerable_recipe_count,
                        aliases,
                    ),
                    "lowerable_recipes": lowerable_recipe_count,
                    "unresolved_bindings": sorted(unresolved),
                })
            record = {
                "candidate_id": candidate.candidate_id,
                "relative_path": relative,
                "qualname": candidate.qualname,
                "start_line": candidate.start_line,
                "end_line": candidate.end_line,
                "kind": candidate.kind,
                "role": roles[candidate.candidate_id],
                "capsule_path": capsule_relative,
                "manifest_path": manifest_relative,
                "recipe_paths": recipe_relatives,
                "lowerable_recipes": lowerable_recipe_count,
            }
            all_records.append(record)
            totals[f"kind:{candidate.kind}"] += 1
            totals[f"role:{roles[candidate.candidate_id]}"] += 1
    known_ids = {item["candidate_id"] for item in all_records}
    orphan_overlays = sorted(set(overlays_by_candidate) - known_ids)
    if orphan_overlays:
        raise RuntimeError(f"curated recipes reference unknown {dsl} candidates: {orphan_overlays}")
    all_records.sort(key=lambda item: (
        item["relative_path"], item["start_line"], item["qualname"], item["candidate_id"]
    ))
    index = {
        "schema_version": SOURCE_INDEX_VERSION,
        "tool_version": TOOL_VERSION,
        "dsl": dsl,
        "upstream": {
            "expected_commit": DSL_LOCKS[dsl]["commit"],
            "actual_commit": actual_commit,
            "backend_commit": DSL_LOCKS[dsl]["backend_commit"],
        },
        "include_prefixes": list(DEFAULT_INCLUDE[dsl]),
        "counts": dict(sorted(totals.items())),
        "candidates": all_records,
    }
    atomic_write(corpus_root / "source-index.json", pretty_json(index))
    if dsl == "triton" and triton_rejection_report is not None:
        structural_counts = Counter(item["bucket"] for item in triton_candidates)
        attempt_counts = Counter(item["code"] for item in triton_attempts)
        rejected_counts = Counter(
            item["code"] for item in triton_attempts if item["status"] == "REJECTED"
        )
        report = {
            "schema_version": "pipeguard-triton-rejection-report-v1",
            "upstream_commit": actual_commit,
            "counts": {
                "structural_buckets": dict(sorted(structural_counts.items())),
                "attempt_codes": dict(sorted(attempt_counts.items())),
                "rejection_codes": dict(sorted(rejected_counts.items())),
            },
            "candidates": sorted(
                triton_candidates,
                key=lambda item: (
                    item["relative_path"], item["qualname"], item["candidate_id"]
                ),
            ),
            "attempts": sorted(
                triton_attempts,
                key=lambda item: (
                    item["relative_path"], item["line"], item["end_line"], item["mechanism"]
                ),
            ),
        }
        atomic_write(triton_rejection_report, pretty_json(report))
    return index


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract locked upstream TileLang/Triton kernels into auditable Python capsules")
    parser.add_argument("--dsl", choices=("tilelang", "triton", "all"), default="all")
    parser.add_argument("--tilelang-root", type=Path)
    parser.add_argument("--triton-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-commit-mismatch", action="store_true")
    parser.add_argument("--triton-rejection-report", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    selected = ("tilelang", "triton") if arguments.dsl == "all" else (arguments.dsl,)
    roots = {"tilelang": arguments.tilelang_root, "triton": arguments.triton_root}
    summaries: dict[str, Any] = {}
    for dsl in selected:
        root = roots[dsl]
        if root is None:
            raise SystemExit(f"--{dsl}-root is required for --dsl {arguments.dsl}")
        index = extract_dsl(
            dsl,
            root,
            arguments.output,
            allow_commit_mismatch=arguments.allow_commit_mismatch,
            triton_rejection_report=(
                arguments.triton_rejection_report if dsl == "triton" else None
            ),
        )
        summaries[dsl] = index["counts"]
    print(canonical_json({"extracted": summaries, "tool_version": TOOL_VERSION}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
