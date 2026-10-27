#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from upstream_common import atomic_write, pretty_json, stable_id
from upstream_flaggems_specs import _test_cases


SCHEMA_VERSION = "pipeguard-tilelang-library-specification-audit-v1"
_UNKNOWN = object()
_MISSING_BINDING = {"__pipeguard_missing__": True}
_IGNORED_BOUNDARY_KEYS = {"label", "operator", "manifest", "row_index"}
_FEATURE_PATTERNS = {
    "PIPELINED": ("T.Pipelined", ".Pipelined("),
    "MBARRIER": ("mbarrier", "MBarrier"),
    "TMA": ("TMA", "tma", "T.copy("),
    "WGMMA": ("T.gemm(", "gemm_ss", "warpgroup"),
    "WARP_SPECIALIZATION": ("warp_special", "num_consumer_groups", "consumer_warp"),
}


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha(path: Path) -> str:
    return _sha_bytes(path.read_bytes())


def _git_files(root: Path, *patterns: str) -> list[str]:
    process = subprocess.run(
        ["git", "-C", str(root), "ls-files", *patterns],
        check=True,
        capture_output=True,
        text=True,
    )
    return sorted(set(process.stdout.splitlines()))


def _git_commit(root: Path) -> str:
    process = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return process.stdout.strip()


def _json_key(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _qualified(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _qualified(node.value)
        return f"{parent}.{node.attr}" if parent else None
    if isinstance(node, ast.Call):
        return _qualified(node.func)
    return None


def _called_names(node: ast.AST) -> set[str]:
    result: set[str] = set()
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        name = _qualified(child.func)
        if name:
            result.add(name.split(".")[-1])
    return result


def _feature_codes(text: str) -> list[str]:
    return sorted(
        code for code, patterns in _FEATURE_PATTERNS.items()
        if any(pattern in text for pattern in patterns)
    )


def _candidate_text(source_root: Path, manifest: dict[str, Any]) -> str:
    upstream = manifest["upstream"]
    lines = (source_root / upstream["relative_path"]).read_text(encoding="utf-8").splitlines(keepends=True)
    return "".join(lines[upstream["start_line"] - 1:upstream["end_line"]])


def _diff_dimensions(original: dict[str, Any], candidate: dict[str, Any]) -> list[str]:
    keys = sorted((set(original) | set(candidate)) - _IGNORED_BOUNDARY_KEYS)
    return [key for key in keys if original.get(key, _UNKNOWN) != candidate.get(key, _UNKNOWN)]


def _select_matrix(cases: Iterable[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    materialized = list(cases)
    binding_keys = sorted({key for case in materialized for key in case["bindings"]})
    normalized_cases = [
        {
            **case,
            "bindings": {
                key: case["bindings"].get(key, _MISSING_BINDING)
                for key in binding_keys
            },
        }
        for case in materialized
    ]
    unique: dict[str, dict[str, Any]] = {}
    for case in normalized_cases:
        unique.setdefault(_json_key(case["bindings"]), case)
    ordered = sorted(
        unique.values(),
        key=lambda item: (
            item["evidence_path"], item["evidence_function"], item["representative_index"],
            item["case_id"],
        ),
    )
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
                item["representative_index"], item["case_id"],
            ),
        )
        chosen = ranked[0]
        dimensions = _diff_dimensions(original["bindings"], chosen["bindings"])
        remaining = [item for item in remaining if item["case_id"] != chosen["case_id"]]
        if not dimensions:
            continue
        selected.append({**chosen, "boundary_dimensions": dimensions})
        covered.update(dimensions)
    return original, selected, sorted(covered)


def _best_matrix(cases: Iterable[dict[str, Any]]) -> tuple[dict[str, Any] | None, list[dict[str, Any]], list[str]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        grouped[(case["evidence_path"], case["evidence_function"])].append(case)
    attempts = [_select_matrix(grouped[key]) for key in sorted(grouped)]
    complete = [item for item in attempts if len(item[1]) >= 3 and len(item[2]) >= 2]
    if complete:
        return complete[0]
    if not attempts:
        return None, [], []
    return max(attempts, key=lambda item: (len(item[1]), len(item[2])))


def _yaml_entry_spans(path: Path) -> dict[str, tuple[int, int, str]]:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    starts: list[tuple[str, int]] = []
    for index, line in enumerate(lines, 1):
        if line and not line[0].isspace() and line.rstrip().endswith(":") and not line.startswith("#"):
            starts.append((line.rstrip()[:-1], index))
    result: dict[str, tuple[int, int, str]] = {}
    for position, (name, start) in enumerate(starts):
        end = (starts[position + 1][1] - 1) if position + 1 < len(starts) else len(lines)
        payload = "".join(lines[start - 1:end])
        result[name] = (start, end, _sha_bytes(payload.encode()))
    return result


def _tileops_module_map(
    source_root: Path,
    relative_path: str,
    candidates: list[dict[str, Any]],
    class_names: set[str],
) -> set[str]:
    path = source_root / relative_path
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
    functions = {
        node.name: node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    classes = {node.name: node for node in tree.body if isinstance(node, ast.ClassDef)}
    roots = {candidate["qualname"].split(".")[0]: candidate["candidate_id"] for candidate in candidates}
    pending: list[str] = []
    for class_name in sorted(class_names):
        node = classes.get(class_name)
        if node is not None:
            pending.extend(_called_names(node))
    reached: set[str] = set()
    while pending:
        name = pending.pop()
        if name in reached:
            continue
        reached.add(name)
        function = functions.get(name)
        if function is not None:
            pending.extend(_called_names(function))
    matched = {candidate_id for name, candidate_id in roots.items() if name in reached}
    if not matched and len(roots) == 1 and class_names:
        matched.update(roots.values())
    return matched


def _expand_tileops_workloads(
    op_name: str,
    manifest_path: str,
    spec: dict[str, Any],
    span: tuple[int, int, str],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_index, raw in enumerate(spec.get("workloads") or []):
        if not isinstance(raw, dict):
            continue
        dtype_values = raw.get("dtypes")
        variants = dtype_values if isinstance(dtype_values, list) and dtype_values else [None]
        for variant_index, dtype in enumerate(variants):
            bindings = {str(key): value for key, value in raw.items() if key != "dtypes"}
            if dtype is not None:
                bindings["dtype"] = dtype
            bindings["operator"] = op_name
            bindings["manifest"] = manifest_path
            bindings["row_index"] = row_index
            case_id = "manifest-" + stable_id(
                manifest_path, op_name, row_index, variant_index, bindings
            )
            rows.append({
                "case_id": case_id,
                "evidence_kind": "MANIFEST_WORKLOAD",
                "evidence_path": manifest_path,
                "evidence_function": op_name,
                "evidence_start_line": span[0],
                "evidence_end_line": span[1],
                "evidence_source_sha256": span[2],
                "bindings": bindings,
                "representative_index": len(rows),
            })
    return rows


def _tileops_cases(
    source_root: Path,
    index: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]], dict[str, int]]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("TileOPs specification extraction requires PyYAML") from error
    jit_by_path: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for candidate in index["candidates"]:
        if candidate["kind"] == "JIT_ENTRY":
            jit_by_path[candidate["relative_path"]].append(candidate)
    cases_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    operators_by_candidate: dict[str, set[str]] = defaultdict(set)
    counts = Counter()
    for relative in _git_files(source_root, "src/tileops/manifest/*.yaml"):
        path = source_root / relative
        document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        spans = _yaml_entry_spans(path)
        for op_name, spec in document.items():
            if not isinstance(spec, dict):
                continue
            counts["manifest_operators"] += 1
            source = spec.get("source") or {}
            kernel_paths = source.get("kernel") or []
            if isinstance(kernel_paths, str):
                kernel_paths = [kernel_paths]
            rows = _expand_tileops_workloads(op_name, relative, spec, spans[op_name])
            counts["manifest_workload_rows"] += len(spec.get("workloads") or [])
            counts["expanded_workload_rows"] += len(rows)
            matched: set[str] = set()
            for kernel_path in kernel_paths:
                normalized = kernel_path if kernel_path.startswith("src/") else f"src/{kernel_path}"
                candidates = jit_by_path.get(normalized, [])
                class_names = set((source.get("kernel_map") or {}).values())
                matched.update(_tileops_module_map(source_root, normalized, candidates, class_names))
            if matched:
                counts["operators_mapped_to_jit_roots"] += 1
            elif rows:
                counts["operators_with_unmapped_workloads"] += 1
            for candidate_id in matched:
                operators_by_candidate[candidate_id].add(op_name)
                cases_by_candidate[candidate_id].extend(rows)
    return cases_by_candidate, operators_by_candidate, dict(counts)


def _normalized_test_stem(path: str) -> str:
    stem = Path(path).stem
    if stem.startswith("test_"):
        stem = stem[5:]
    return stem.replace("topk_sum_and_topk_idx", "topk_sum_and_topk_group_idx")


def _normalized_kernel_stem(path: str) -> str:
    stem = Path(path).stem
    return stem[:-7] if stem.endswith("_kernel") else stem


def _tilekernels_cases(
    source_root: Path,
    index: dict[str, Any],
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, set[str]], dict[str, int]]:
    tests = _test_cases(source_root)
    tests_by_family_stem: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for case in tests:
        parts = Path(case["path"]).parts
        family = parts[1] if len(parts) > 2 and parts[0] == "tests" else ""
        tests_by_family_stem[(family, _normalized_test_stem(case["path"]))].append(case)
    all_test_paths = _git_files(source_root, "tests/*.py", "tests/**/*.py")
    test_files = {
        ((Path(path).parts[1] if len(Path(path).parts) > 2 else ""), _normalized_test_stem(path))
        for path in all_test_paths
        if Path(path).name.startswith("test_")
    }
    cases_by_candidate: dict[str, list[dict[str, Any]]] = defaultdict(list)
    tests_by_candidate: dict[str, set[str]] = defaultdict(set)
    unresolved_test_matrix = 0
    for candidate in index["candidates"]:
        if candidate["kind"] != "JIT_ENTRY":
            continue
        parts = Path(candidate["relative_path"]).parts
        family = parts[1] if len(parts) > 2 and parts[0] == "tile_kernels" else ""
        stem = _normalized_kernel_stem(candidate["relative_path"])
        matched = tests_by_family_stem.get((family, stem), [])
        for case in matched:
            tests_by_candidate[candidate["candidate_id"]].add(case["path"])
            cases_by_candidate[candidate["candidate_id"]].append({
                "case_id": case["test_case_id"],
                "evidence_kind": "UPSTREAM_TEST",
                "evidence_path": case["path"],
                "evidence_function": case["function"],
                "evidence_start_line": case["start_line"],
                "evidence_end_line": case["end_line"],
                "evidence_source_sha256": case["source_sha256"],
                "bindings": case["bindings"],
                "representative_index": case["representative_index"],
            })
        if not matched and (family, stem) in test_files:
            unresolved_test_matrix += 1
            tests_by_candidate[candidate["candidate_id"]].add(f"tests/{family}/test_{stem}.py")
    return cases_by_candidate, tests_by_candidate, {
        "official_test_representatives": len(tests),
        "test_files": len(all_test_paths),
        "roots_with_unresolved_test_matrix": unresolved_test_matrix,
    }


def extract_specifications(
    *,
    project: str,
    source_id: str,
    repository_url: str,
    source_root: Path,
    source_index_path: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve(strict=True)
    source_index_path = source_index_path.resolve(strict=True)
    index = _load(source_index_path)
    commit = _git_commit(source_root)
    if commit != index["upstream"]["actual_commit"]:
        raise RuntimeError(
            f"source checkout mismatch: index={index['upstream']['actual_commit']} actual={commit}"
        )
    if project == "tileops":
        cases_by_candidate, names_by_candidate, project_counts = _tileops_cases(source_root, index)
        evidence_label = "operator_names"
    elif project == "tilekernels":
        cases_by_candidate, names_by_candidate, project_counts = _tilekernels_cases(source_root, index)
        evidence_label = "test_paths"
    else:
        raise ValueError(f"unknown project mode: {project}")

    manifests = {
        candidate["candidate_id"]: _load(source_index_path.parent.parent / candidate["manifest_path"])
        for candidate in index["candidates"]
    }
    primfuncs = [candidate for candidate in index["candidates"] if candidate["kind"] == "PRIMFUNC"]
    roots: list[dict[str, Any]] = []
    specifications: list[dict[str, Any]] = []
    for candidate in index["candidates"]:
        if candidate["kind"] != "JIT_ENTRY":
            continue
        candidate_id = candidate["candidate_id"]
        manifest = manifests[candidate_id]
        implementation_ids = sorted(
            item["candidate_id"] for item in primfuncs
            if item["relative_path"] == candidate["relative_path"]
            and item["qualname"].startswith(candidate["qualname"] + ".")
        )
        text = _candidate_text(source_root, manifest)
        cases = cases_by_candidate.get(candidate_id, [])
        original, boundaries, dimensions = _best_matrix(cases)
        if original is None:
            status = (
                "UNRESOLVED_TEST_MATRIX"
                if project == "tilekernels" and names_by_candidate.get(candidate_id)
                else "NO_OFFICIAL_SPECIFICATION"
            )
        elif len(boundaries) < 3 or len(dimensions) < 2:
            status = "INSUFFICIENT_BOUNDARY_EVIDENCE"
        else:
            status = "SOURCE_REPLAY_READY"
        root_id = f"{source_id}-root-{candidate_id}"
        original_id: str | None = None
        boundary_ids: list[str] = []
        selected = [] if original is None else [("ORIGINAL", original)] + [
            ("BOUNDARY", item) for item in boundaries
        ]
        for role, case in selected:
            suffix = stable_id(case["case_id"], case["bindings"], length=10)
            spec_id = f"{root_id}-{role.lower()}-{suffix}"
            if role == "ORIGINAL":
                original_id = spec_id
            else:
                boundary_ids.append(spec_id)
            specifications.append({
                "spec_id": spec_id,
                "root_id": root_id,
                "candidate_id": candidate_id,
                "role": role,
                "parent_spec_id": None if role == "ORIGINAL" else original_id,
                "evidence_kind": case["evidence_kind"],
                "evidence_path": case["evidence_path"],
                "evidence_function": case["evidence_function"],
                "evidence_start_line": case["evidence_start_line"],
                "evidence_end_line": case["evidence_end_line"],
                "evidence_source_sha256": case["evidence_source_sha256"],
                "bindings": case["bindings"],
                "boundary_dimensions": [] if role == "ORIGINAL" else case["boundary_dimensions"],
                "materialization_status": "SOURCE_SPECIFICATION_ONLY",
                "expected_pipeline": "UNASSESSED",
            })
        roots.append({
            "root_id": root_id,
            "candidate_id": candidate_id,
            "path": candidate["relative_path"],
            "qualname": candidate["qualname"],
            "source_role": candidate["role"],
            "implementation_candidate_ids": implementation_ids,
            "feature_codes": _feature_codes(text),
            "status": status,
            evidence_label: sorted(names_by_candidate.get(candidate_id, set())),
            "matched_specification_rows": len(cases),
            "original_spec_id": original_id,
            "boundary_spec_ids": boundary_ids,
            "boundary_dimensions": dimensions,
        })

    status_counts = Counter(item["status"] for item in roots)
    feature_counts = Counter(code for item in roots for code in item["feature_codes"])
    counts = {
        "source_files": index["counts"].get("source_files", 0),
        "source_candidates": index["counts"].get("candidates", 0),
        "jit_roots": len(roots),
        "primfunc_definitions": index["counts"].get("kind:PRIMFUNC", 0),
        "root_status": dict(sorted(status_counts.items())),
        "feature_roots": dict(sorted(feature_counts.items())),
        "original_specifications": sum(item["role"] == "ORIGINAL" for item in specifications),
        "boundary_specifications": sum(item["role"] == "BOUNDARY" for item in specifications),
        "source_replay_ready_roots": status_counts["SOURCE_REPLAY_READY"],
        "native_materialized_roots": 0,
        "safe_roots": 0,
        **project_counts,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "source_id": source_id,
        "project": "TileOPs" if project == "tileops" else "TileKernels",
        "repository_url": repository_url,
        "commit": commit,
        "source_index": {
            "path": source_index_path.name,
            "sha256": _sha(source_index_path),
        },
        "counts": counts,
        "roots": sorted(roots, key=lambda item: item["root_id"]),
        "specifications": sorted(specifications, key=lambda item: item["spec_id"]),
    }


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Recover source-grounded specifications for external TileLang libraries"
    )
    parser.add_argument("--project", choices=("tileops", "tilekernels"), required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--repository-url", required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--source-index", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    result = extract_specifications(
        project=arguments.project,
        source_id=arguments.source_id,
        repository_url=arguments.repository_url,
        source_root=arguments.source_root,
        source_index_path=arguments.source_index,
    )
    atomic_write(arguments.output, pretty_json(result))
    print(json.dumps(result["counts"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
