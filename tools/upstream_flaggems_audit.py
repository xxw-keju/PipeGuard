#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from upstream_common import atomic_write, pretty_json, stable_id


SCHEMA_VERSION = "pipeguard-flaggems-launch-root-audit-v1"

_FEATURE_PATTERNS = {
    "TLE": (
        "flagtree",
        "\ntle.",
        " tle.",
        "@tle.",
        "_tle_",
    ),
    "WARP_SPECIALIZATION": (
        "warp_specialize",
        "num_consumer_groups",
        "num_buffers_warp_spec",
    ),
    "DESCRIPTOR_TMA": (
        "tensor_descriptor",
        "tensormap",
        "descriptor_load",
        "descriptor_store",
        "tma_",
    ),
    "EXTERNAL_DEVICE": (
        "libdevice",
        "extern_elementwise",
        "external_libs",
    ),
}

_STATUS_PRECEDENCE = (
    ("TLE", "EXCLUDED_TLE"),
    ("WARP_SPECIALIZATION", "EXCLUDED_UNSUPPORTED_WARP_SPECIALIZATION"),
    ("DESCRIPTOR_TMA", "EXCLUDED_UNSUPPORTED_DESCRIPTOR_TMA"),
    ("EXTERNAL_DEVICE", "EXCLUDED_EXTERNAL_DEVICE_DEPENDENCY"),
)


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _source_span(source_lines: list[str], node: ast.AST) -> str:
    start = getattr(node, "lineno", 1)
    end = getattr(node, "end_lineno", start)
    return "".join(source_lines[start - 1:end])


def _safe_unparse(node: ast.AST) -> str:
    try:
        return ast.unparse(node)
    except Exception:
        return "<unparseable>"


def _launch_target(call: ast.Call) -> tuple[ast.AST | None, str]:
    if isinstance(call.func, ast.Subscript):
        return call.func.value, "BRACKET"
    if isinstance(call.func, ast.Attribute) and call.func.attr in {"warmup", "compile"}:
        return call.func.value, call.func.attr.upper()
    return None, ""


def _target_name(node: ast.AST | None) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


class LaunchVisitor(ast.NodeVisitor):
    def __init__(
        self,
        *,
        source_id: str,
        relative_path: str,
        source: str,
        candidates: list[dict[str, Any]],
    ):
        self.source_id = source_id
        self.relative_path = relative_path
        self.source_lines = source.splitlines(keepends=True)
        self.scope: list[str] = []
        self.branches: list[str] = []
        self.ordinals: Counter[str] = Counter()
        self.occurrences: list[dict[str, Any]] = []
        self.candidates_by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for candidate in candidates:
            self.candidates_by_name[candidate["qualname"].split(".")[-1]].append(candidate)

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.scope.append(node.name)
        for statement in node.body:
            self.visit(statement)
        self.scope.pop()

    def visit_If(self, node: ast.If) -> None:
        condition = _safe_unparse(node.test)
        self.visit(node.test)
        self.branches.append(f"IF:{condition}")
        for statement in node.body:
            self.visit(statement)
        self.branches.pop()
        if node.orelse:
            self.branches.append(f"ELSE:{condition}")
            for statement in node.orelse:
                self.visit(statement)
            self.branches.pop()

    def visit_Call(self, node: ast.Call) -> None:
        target, mechanism = _launch_target(node)
        if target is not None:
            self._record(node, target, mechanism)
        self.generic_visit(node)

    def _resolve(self, target: ast.AST) -> tuple[str, str | None, list[str]]:
        name = _target_name(target)
        if name is None:
            return "UNRESOLVED", None, []
        candidates = self.candidates_by_name.get(name, [])
        if not candidates:
            return "UNRESOLVED", None, []
        visible: list[tuple[int, dict[str, Any]]] = []
        scope = tuple(self.scope)
        for candidate in candidates:
            parents = tuple(candidate["lexical_parents"])
            if len(parents) <= len(scope) and scope[:len(parents)] == parents:
                visible.append((len(parents), candidate))
        if not visible:
            return "UNRESOLVED", None, sorted(item["candidate_id"] for item in candidates)
        depth = max(item[0] for item in visible)
        nearest = [item for item_depth, item in visible if item_depth == depth]
        if len(nearest) != 1:
            return "AMBIGUOUS", None, sorted(item["candidate_id"] for item in nearest)
        return "RESOLVED", nearest[0]["candidate_id"], []

    def _record(self, node: ast.Call, target: ast.AST, mechanism: str) -> None:
        host_qualname = ".".join(self.scope) or "<module>"
        ordinal = self.ordinals[host_qualname]
        self.ordinals[host_qualname] += 1
        resolution, candidate_id, alternatives = self._resolve(target)
        target_text = _safe_unparse(target)
        occurrence_id = f"launch-{stable_id(self.source_id, self.relative_path, node.lineno, node.end_lineno or node.lineno, host_qualname, ordinal, target_text)}"
        code = None
        if resolution == "UNRESOLVED":
            code = "TARGET_NOT_LEXICALLY_RESOLVED"
        elif resolution == "AMBIGUOUS":
            code = "TARGET_AMBIGUOUS"
        self.occurrences.append({
            "occurrence_id": occurrence_id,
            "path": self.relative_path,
            "start_line": node.lineno,
            "end_line": node.end_lineno or node.lineno,
            "source_sha256": _sha256_bytes(_source_span(self.source_lines, node).encode("utf-8")),
            "host_qualname": host_qualname,
            "launch_group_id": f"group-{stable_id(self.source_id, self.relative_path, host_qualname)}",
            "launch_ordinal": ordinal,
            "mechanism": mechanism,
            "target": target_text,
            "branch_predicates": list(self.branches),
            "resolution": resolution,
            "diagnostic_code": code,
            "candidate_id": candidate_id,
            "candidate_alternatives": alternatives,
        })


def _candidate_text(source_root: Path, manifest: dict[str, Any]) -> str:
    upstream = manifest["upstream"]
    lines = (source_root / upstream["relative_path"]).read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(lines[upstream["start_line"] - 1:upstream["end_line"]])


def _feature_codes(
    candidate_id: str,
    *,
    manifests: dict[str, dict[str, Any]],
    source_root: Path,
) -> list[str]:
    pending = [candidate_id]
    seen: set[str] = set()
    text: list[str] = []
    while pending:
        current = pending.pop()
        if current in seen:
            continue
        seen.add(current)
        manifest = manifests.get(current)
        if manifest is None:
            continue
        text.append(_candidate_text(source_root, manifest).lower())
        pending.extend(manifest.get("kernel_dependencies", []))
    joined = "\n".join(text)
    return sorted(
        feature
        for feature, patterns in _FEATURE_PATTERNS.items()
        if any(pattern in joined for pattern in patterns)
    )


def _audit_status(features: list[str]) -> str:
    for feature, status in _STATUS_PRECEDENCE:
        if feature in features:
            return status
    return "READY_FOR_RECIPE_RECOVERY"


def audit_flaggems(
    *,
    source_root: Path,
    corpus_root: Path,
    source_index_path: Path,
    source_id: str,
    repository_url: str,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    corpus_root = corpus_root.resolve(strict=True)
    source_index_path = source_index_path.resolve(strict=True)
    index = _load(source_index_path)
    commit = index["upstream"]["actual_commit"]
    candidates = {item["candidate_id"]: item for item in index["candidates"]}
    manifests = {
        candidate_id: _load(corpus_root / candidate["manifest_path"])
        for candidate_id, candidate in candidates.items()
    }
    by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate_id, candidate in candidates.items():
        manifest = manifests[candidate_id]
        by_path[candidate["relative_path"]].append({
            **candidate,
            "lexical_parents": manifest.get("lexical_parents", []),
        })

    occurrences: list[dict[str, Any]] = []
    for relative_path in sorted(by_path):
        source = (source_root / relative_path).read_text(encoding="utf-8")
        tree = ast.parse(source, filename=relative_path)
        visitor = LaunchVisitor(
            source_id=source_id,
            relative_path=relative_path,
            source=source,
            candidates=by_path[relative_path],
        )
        visitor.visit(tree)
        occurrences.extend(visitor.occurrences)
    occurrences.sort(key=lambda item: (
        item["path"], item["start_line"], item["end_line"], item["occurrence_id"]
    ))

    resolved_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for occurrence in occurrences:
        if occurrence["candidate_id"] is not None:
            resolved_by_candidate[occurrence["candidate_id"]].append(occurrence)

    prior_roots = {
        candidate_id for candidate_id, candidate in candidates.items()
        if candidate["role"] in {"ROOT", "ROOT_HELPER"}
    }
    candidate_results: list[dict[str, Any]] = []
    roots: list[dict[str, Any]] = []
    for candidate_id in sorted(prior_roots):
        candidate = candidates[candidate_id]
        resolved = resolved_by_candidate.get(candidate_id, [])
        if not resolved:
            result = "NOT_A_LAUNCH_ROOT"
            reason = "NO_SCOPE_RESOLVED_LAUNCH"
        else:
            result = "TRUE_LAUNCH_ROOT"
            reason = None
        candidate_results.append({
            "candidate_id": candidate_id,
            "previous_role": candidate["role"],
            "audit_result": result,
            "diagnostic_code": reason,
            "resolved_launch_occurrence_ids": [item["occurrence_id"] for item in resolved],
        })
        if not resolved:
            continue
        features = _feature_codes(
            candidate_id, manifests=manifests, source_root=source_root
        )
        roots.append({
            "root_id": f"flaggems-root-{candidate_id}",
            "candidate_id": candidate_id,
            "path": candidate["relative_path"],
            "qualname": candidate["qualname"],
            "kind": candidate["kind"],
            "source_role": candidate["role"],
            "launch_occurrence_ids": [item["occurrence_id"] for item in resolved],
            "launch_group_ids": sorted({item["launch_group_id"] for item in resolved}),
            "feature_codes": features,
            "audit_status": _audit_status(features),
            "recipe_status": "NOT_ATTEMPTED",
            "recipe_ids": [],
            "boundary_status": "NOT_ATTEMPTED",
            "boundary_spec_ids": [],
        })

    resolution_counts = Counter(item["resolution"] for item in occurrences)
    status_counts = Counter(item["audit_status"] for item in roots)
    result_counts = Counter(item["audit_result"] for item in candidate_results)
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "repository_url": repository_url,
        "commit": commit,
        "source_index": {
            "path": source_index_path.relative_to(corpus_root).as_posix(),
            "sha256": _sha256_bytes(source_index_path.read_bytes()),
        },
        "counts": {
            "source_candidates": len(candidates),
            "previous_root_like": len(prior_roots),
            "launch_occurrences": len(occurrences),
            "resolution": dict(sorted(resolution_counts.items())),
            "candidate_audit": dict(sorted(result_counts.items())),
            "true_launch_roots": len(roots),
            "audit_status": dict(sorted(status_counts.items())),
        },
        "launch_occurrences": occurrences,
        "candidate_audit": candidate_results,
        "roots": sorted(roots, key=lambda item: item["root_id"]),
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit FlagGems launch roots without importing or executing kernels")
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--corpus-root", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--source-id", default="flaggems-34bd6d68")
    parser.add_argument("--repository-url", default="https://github.com/flagos-ai/FlagGems")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    result = audit_flaggems(
        source_root=arguments.source_root,
        corpus_root=arguments.corpus_root,
        source_index_path=arguments.source_index,
        source_id=arguments.source_id,
        repository_url=arguments.repository_url,
    )
    atomic_write(arguments.output, pretty_json(result))
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
