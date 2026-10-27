#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from upstream_common import atomic_write, pretty_json, safe_component, sha256_bytes, stable_id
from upstream_suite import (
    _diff_paths,
    _normalized_recipe_for_delta,
    _pointer_get,
    validate_suite,
)


_ORIGIN_LOCATION = re.compile(r"(?P<path>(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+\.py):(?P<start>[0-9]+)(?:-(?P<end>[0-9]+))?")


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _source_range_hash(path: Path, start: int, end: int) -> str:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    if start < 1 or end < start or end > len(lines):
        raise ValueError(f"invalid source range {path}:{start}-{end}")
    return sha256_bytes("".join(lines[start - 1:end]).encode("utf-8"))


def _permalink(repository_url: str, commit: str, path: str, start: int, end: int) -> str:
    return f"{repository_url}/blob/{commit}/{path}#L{start}-L{end}"


def _launch_evidence(
    *,
    recipe: dict[str, Any],
    repository: dict[str, Any],
    source_root: Path,
) -> dict[str, Any] | None:
    match = _ORIGIN_LOCATION.search(recipe["origin"])
    if match is None:
        return None
    relative_path = match.group("path")
    start = int(match.group("start"))
    end = int(match.group("end") or start)
    path = source_root / relative_path
    if not path.is_file():
        return None
    evidence_id = f"launch-{stable_id(repository['source_id'], relative_path, start, end)}"
    return {
        "evidence_id": evidence_id,
        "kind": "UPSTREAM_TEST" if "/test" in f"/{relative_path}" or relative_path.startswith("testing/") else "LAUNCH_CALLSITE",
        "repository_id": repository["source_id"],
        "commit": repository["commit"],
        "path": relative_path,
        "start_line": start,
        "end_line": end,
        "url": _permalink(repository["repository_url"], repository["commit"], relative_path, start, end),
        "sha256": _source_range_hash(path, start, end),
        "rationale": f"Concrete specialization origin recorded by upstream recipe {recipe['recipe_id']}: {recipe['origin']}",
    }


def _source_evidence(
    *,
    manifest: dict[str, Any],
    repository: dict[str, Any],
) -> dict[str, Any]:
    upstream = manifest["upstream"]
    evidence_id = f"source-{manifest['candidate_id']}"
    return {
        "evidence_id": evidence_id,
        "kind": "KERNEL_SOURCE",
        "repository_id": repository["source_id"],
        "commit": repository["commit"],
        "path": upstream["relative_path"],
        "start_line": upstream["start_line"],
        "end_line": upstream["end_line"],
        "url": _permalink(
            repository["repository_url"], repository["commit"], upstream["relative_path"],
            upstream["start_line"], upstream["end_line"],
        ),
        "sha256": upstream["selected_source_sha256"],
        "rationale": "Pinned source span defining the extracted device candidate.",
    }


def _claim_scope(recipe: dict[str, Any]) -> str:
    compiler = recipe.get("compiler", {})
    grid = compiler.get("grid")
    if grid is not None:
        count = 1
        for value in grid:
            count *= value
    else:
        count = compiler.get("grid_ctas", 1)
    return "SINGLE_CTA" if count == 1 else "SELECTED_CTA"


def _kernel_id(dsl: str, candidate: dict[str, Any]) -> str:
    name = safe_component(candidate["qualname"].replace(".", "-"), maximum=48).lower()
    return f"{dsl}-{name}-{candidate['candidate_id'][:8]}"


def _candidate_reason(candidate: dict[str, Any]) -> str:
    if candidate["lowerable_recipes"] == 0:
        return "Source extraction succeeded, but no concrete original launch recipe is currently recoverable; materialization and boundary review remain pending."
    role = candidate["role"]
    if role in {"ROOT", "ROOT_HELPER"}:
        return "Original recipe is locked, but three source-grounded boundary specifications across two dimensions have not yet been reviewed."
    if role == "HELPER":
        return "Extractable device helper retained for coverage accounting; direct-launch status and independent boundary specifications require review."
    return "Extractable device candidate retained for coverage accounting; launch-root classification and independent boundary specifications require review."


def _flaggems_replay_recipe(
    spec: dict[str, Any],
) -> dict[str, Any]:
    evidence = {
        "kind": "UPSTREAM_TEST_REPLAY",
        "test_path": spec["test_path"],
        "test_function": spec["test_function"],
        "test_start_line": spec["test_start_line"],
        "test_end_line": spec["test_end_line"],
        "test_source_sha256": spec["test_source_sha256"],
        "launch_occurrence_ids": spec["launch_occurrence_ids"],
    }
    return {
        "schema_version": "pipeguard-upstream-recipe-v1",
        "recipe_id": f"{spec['spec_id']}-recipe",
        "candidate_id": spec["candidate_id"],
        "dsl": "triton",
        "origin": (
            f"upstream-test:{spec['test_path']}:"
            f"{spec['test_start_line']}-{spec['test_end_line']}#"
            f"{spec['test_function']}"
        ),
        "lowerable": False,
        "unresolved": [
            "flaggems_triton_3_6_toolchain",
            "native_test_replay_materialization",
        ],
        "bindings": spec["bindings"],
        "compiler": {
            "selected_cta": [0, 0, 0],
            "binding_evidence": evidence,
        },
        "reference": {"mode": "not_run"},
        "target": {"backend": "cuda", "arch": "sm_90a", "warp_size": 32},
    }


def _pointer_get_or_missing(value: Any, pointer: str) -> Any:
    try:
        return _pointer_get(value, pointer)
    except ValueError:
        return {"__pipeguard_missing__": True}


def _recipe_delta(
    parent: dict[str, Any],
    child: dict[str, Any],
) -> list[dict[str, Any]]:
    parent_normalized = _normalized_recipe_for_delta(parent)
    child_normalized = _normalized_recipe_for_delta(child)
    return [
        {
            "path": pointer,
            "before": _pointer_get_or_missing(parent_normalized, pointer),
            "after": _pointer_get_or_missing(child_normalized, pointer),
        }
        for pointer in sorted(_diff_paths(parent_normalized, child_normalized))
    ]


def _merge_flaggems_specification_audit(
    *,
    corpus_root: Path,
    source_root: Path,
    repository: dict[str, Any],
    evidence: dict[str, dict[str, Any]],
    revisions: dict[str, dict[str, Any]],
    kernels: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    engine: str,
) -> None:
    audit_path = corpus_root / "flaggems-34bd6d68" / "specification-audit.json"
    root_audit_path = corpus_root / "flaggems-34bd6d68" / "launch-root-audit.json"
    if not audit_path.is_file() or not root_audit_path.is_file():
        return
    audit = _load(audit_path)
    root_audit = _load(root_audit_path)
    occurrences = {
        item["occurrence_id"]: item for item in root_audit["launch_occurrences"]
    }
    kernel_by_candidate = {
        revision["candidate_id"]: revision["kernel_id"]
        for revision in revisions.values()
        if revision["repository_id"] == repository["source_id"]
    }
    revision_by_candidate = {
        revision["candidate_id"]: revision["revision_id"]
        for revision in revisions.values()
        if revision["repository_id"] == repository["source_id"]
    }
    audited_roots = {item["root_id"]: item for item in audit["roots"]}
    qualified_roots = {
        root_id: item
        for root_id, item in audited_roots.items()
        if item["status"] == "SOURCE_REPLAY_READY"
    }
    for audited_root in audited_roots.values():
        candidate_id = audited_root["candidate_id"]
        kernel = kernels[kernel_by_candidate[candidate_id]]
        kernel["launch_root_rationale"] = (
            "The scope-aware FlagGems audit resolves this device definition to at "
            "least one immutable host launch callsite."
        )
        status = audited_root["status"]
        if status == "SOURCE_REPLAY_READY":
            reason = (
                "Source-replay specifications are complete; native lowering remains "
                "unassessed until the declared Triton 3.6/FlagTree toolchain is pinned."
            )
        elif status == "INSUFFICIENT_BOUNDARY_EVIDENCE":
            reason = (
                "An official original test recipe is present, but fewer than three "
                "boundaries across two dimensions can be established conservatively."
            )
        elif status == "NO_OFFICIAL_TEST_RECIPE":
            reason = (
                "The launch root is source-confirmed, but no exact official-test "
                "replay recipe can currently be matched without guessing."
            )
        else:
            reason = f"Static FlagGems audit stopped this root with status {status}; it was not evaluated for safety."
        kernel["exclusion_reason"] = reason
    selected_spec_ids = {
        spec_id
        for root in qualified_roots.values()
        for spec_id in (root["original_spec_id"], *root["boundary_spec_ids"])
    }
    audit_specs = {
        item["spec_id"]: item
        for item in audit["specifications"]
        if item["spec_id"] in selected_spec_ids
    }
    if len(audit_specs) != 4 * len(qualified_roots):
        raise ValueError("FlagGems specification audit is not a complete 1+3 matrix")

    recipes: dict[str, dict[str, Any]] = {}
    recipe_paths: dict[str, str] = {}
    recipe_root = corpus_root / "suite3" / "recipes" / "flaggems-auto"
    for spec_id in sorted(audit_specs):
        recipe = _flaggems_replay_recipe(audit_specs[spec_id])
        recipe_path = recipe_root / f"{safe_component(spec_id, maximum=180)}.json"
        atomic_write(recipe_path, pretty_json(recipe))
        recipes[spec_id] = recipe
        recipe_paths[spec_id] = recipe_path.relative_to(corpus_root).as_posix()

    for root_id in sorted(qualified_roots):
        audited_root = qualified_roots[root_id]
        candidate_id = audited_root["candidate_id"]
        kernel_id = kernel_by_candidate[candidate_id]
        revision_id = revision_by_candidate[candidate_id]
        kernel = kernels[kernel_id]
        original_id = audited_root["original_spec_id"]
        original_recipe = recipes[original_id]
        kernel["primary_original_spec_id"] = original_id
        kernel["launch_root_rationale"] = (
            "Scope-resolved direct launch root with an immutable official FlagGems "
            "test matrix supplying one original and three boundaries across at least "
            "two dimensions."
        )
        kernel["exclusion_reason"] = (
            "Source-replay specifications are complete; native lowering remains "
            "unassessed until the declared Triton 3.6/FlagTree toolchain is pinned."
        )
        for spec_id in (original_id, *audited_root["boundary_spec_ids"]):
            audited_spec = audit_specs[spec_id]
            recipe = recipes[spec_id]
            test_evidence_id = f"flaggems-test-{stable_id(audited_spec['test_path'], audited_spec['test_function'])}"
            test_path = source_root / audited_spec["test_path"]
            actual_test_hash = _source_range_hash(
                test_path,
                audited_spec["test_start_line"],
                audited_spec["test_end_line"],
            )
            if actual_test_hash != audited_spec["test_source_sha256"]:
                raise ValueError(f"FlagGems test evidence drifted for {spec_id}")
            evidence[test_evidence_id] = {
                "evidence_id": test_evidence_id,
                "kind": "UPSTREAM_TEST",
                "repository_id": repository["source_id"],
                "commit": repository["commit"],
                "path": audited_spec["test_path"],
                "start_line": audited_spec["test_start_line"],
                "end_line": audited_spec["test_end_line"],
                "url": _permalink(
                    repository["repository_url"], repository["commit"],
                    audited_spec["test_path"], audited_spec["test_start_line"],
                    audited_spec["test_end_line"],
                ),
                "sha256": actual_test_hash,
                "rationale": (
                    "Official FlagGems parameter matrix and oracle used as the "
                    "source-replay specification."
                ),
            }
            evidence_ids = [f"source-{candidate_id}", test_evidence_id]
            for occurrence_id in audited_spec["launch_occurrence_ids"]:
                occurrence = occurrences[occurrence_id]
                launch_evidence_id = f"flaggems-{occurrence_id}"
                launch_path = source_root / occurrence["path"]
                actual_launch_hash = _source_range_hash(
                    launch_path, occurrence["start_line"], occurrence["end_line"]
                )
                if actual_launch_hash != occurrence["source_sha256"]:
                    raise ValueError(f"FlagGems launch evidence drifted for {occurrence_id}")
                evidence[launch_evidence_id] = {
                    "evidence_id": launch_evidence_id,
                    "kind": "LAUNCH_CALLSITE",
                    "repository_id": repository["source_id"],
                    "commit": repository["commit"],
                    "path": occurrence["path"],
                    "start_line": occurrence["start_line"],
                    "end_line": occurrence["end_line"],
                    "url": _permalink(
                        repository["repository_url"], repository["commit"],
                        occurrence["path"], occurrence["start_line"],
                        occurrence["end_line"],
                    ),
                    "sha256": actual_launch_hash,
                    "rationale": (
                        f"Scope-resolved {occurrence['mechanism'].lower()} of "
                        f"candidate {candidate_id} from {occurrence['host_qualname']}."
                    ),
                }
                evidence_ids.append(launch_evidence_id)
            role = audited_spec["role"]
            parent_id = audited_spec["parent_spec_id"]
            delta = [] if role == "ORIGINAL" else _recipe_delta(
                original_recipe, recipe
            )
            if role == "BOUNDARY" and not delta:
                raise ValueError(f"FlagGems boundary {spec_id} has no recipe delta")
            item = {
                "spec_id": spec_id,
                "kernel_id": kernel_id,
                "revision_id": revision_id,
                "role": role,
                "recipe": {
                    "path": recipe_paths[spec_id],
                    "recipe_id": recipe["recipe_id"],
                    "candidate_id": candidate_id,
                    "sha256": _hash(corpus_root / recipe_paths[spec_id]),
                },
                "delta": delta,
                "boundary_dimensions": audited_spec["boundary_dimensions"],
                "rule_ids": [],
                "evidence_ids": sorted(set(evidence_ids)),
                "rationale": (
                    "Exact source-replay specialization selected from an official "
                    "FlagGems test parameter matrix; native capture is intentionally "
                    "deferred until the declared toolchain is pinned."
                ),
                "execution": {
                    "selected_cta": [0, 0, 0],
                    "claim_scope": "SELECTED_CTA",
                    "symbol_policy": "ALL",
                    "symbols": [],
                    "engine": engine,
                },
                "expected_pipeline": {
                    "status": "UNASSESSED",
                    "allowed_diagnostic_codes": [],
                },
                "semantic_ground_truth": {
                    "class": "UNKNOWN",
                    "status": "UNADJUDICATED",
                    "method": "UNADJUDICATED",
                    "evidence_ids": [],
                    "rationale": (
                        "The upstream test supplies legal input provenance, but no "
                        "checker result is used as independent semantic ground truth."
                    ),
                },
            }
            if role == "BOUNDARY":
                item["parent_spec_id"] = parent_id
                item["boundary_transition"] = (
                    "Official test-matrix variation across: "
                    + ", ".join(audited_spec["boundary_dimensions"])
                )
            specs[spec_id] = item


def _tilelang_library_replay_recipe(spec: dict[str, Any]) -> dict[str, Any]:
    evidence = {
        "kind": spec["evidence_kind"],
        "path": spec["evidence_path"],
        "function": spec["evidence_function"],
        "start_line": spec["evidence_start_line"],
        "end_line": spec["evidence_end_line"],
        "source_sha256": spec["evidence_source_sha256"],
    }
    return {
        "schema_version": "pipeguard-upstream-recipe-v1",
        "recipe_id": f"{spec['spec_id']}-recipe",
        "candidate_id": spec["candidate_id"],
        "dsl": "tilelang",
        "origin": (
            f"upstream-{spec['evidence_kind'].lower()}:"
            f"{spec['evidence_path']}:"
            f"{spec['evidence_start_line']}-{spec['evidence_end_line']}#"
            f"{spec['evidence_function']}"
        ),
        "lowerable": False,
        "unresolved": [
            "native_factory_materialization",
            "source_specification_to_kernel_binding",
        ],
        "bindings": spec["bindings"],
        "compiler": {
            "selected_cta": [0, 0, 0],
            "binding_evidence": evidence,
        },
        "reference": {"mode": "not_run"},
        "target": {"backend": "cuda", "arch": "sm_90a", "warp_size": 32},
    }


def _merge_tilelang_library_specification_audit(
    *,
    corpus_root: Path,
    source_root: Path,
    repository: dict[str, Any],
    audit_path: Path,
    evidence: dict[str, dict[str, Any]],
    revisions: dict[str, dict[str, Any]],
    kernels: dict[str, dict[str, Any]],
    specs: dict[str, dict[str, Any]],
    engine: str,
) -> None:
    audit = _load(audit_path)
    if audit.get("schema_version") != "pipeguard-tilelang-library-specification-audit-v1":
        raise ValueError(f"invalid TileLang library specification audit: {audit_path}")
    if audit.get("source_id") != repository["source_id"]:
        raise ValueError(f"TileLang audit source mismatch: {audit_path}")
    if audit.get("commit") != repository["commit"]:
        raise ValueError(f"TileLang audit commit mismatch: {audit_path}")

    kernel_by_candidate = {
        revision["candidate_id"]: revision["kernel_id"]
        for revision in revisions.values()
        if revision["repository_id"] == repository["source_id"]
    }
    revision_by_candidate = {
        revision["candidate_id"]: revision["revision_id"]
        for revision in revisions.values()
        if revision["repository_id"] == repository["source_id"]
    }
    audited_roots = {item["root_id"]: item for item in audit["roots"]}
    qualified_roots = {
        root_id: item
        for root_id, item in audited_roots.items()
        if item["status"] == "SOURCE_REPLAY_READY"
    }
    for audited_root in audited_roots.values():
        candidate_id = audited_root["candidate_id"]
        if candidate_id not in kernel_by_candidate:
            raise ValueError(f"TileLang audit references unknown candidate: {candidate_id}")
        kernel = kernels[kernel_by_candidate[candidate_id]]
        kernel["launch_root_rationale"] = (
            f"The pinned {repository['project']} source marks this function with "
            "@tilelang.jit; official manifest/test evidence is audited separately."
        )
        status = audited_root["status"]
        if status == "SOURCE_REPLAY_READY":
            reason = (
                "One source-derived original and three boundaries across at least "
                "two dimensions are recorded; exact factory materialization remains unassessed."
            )
        elif status == "INSUFFICIENT_BOUNDARY_EVIDENCE":
            reason = (
                "Official source evidence exists, but fewer than three boundaries "
                "across two dimensions can be established conservatively."
            )
        elif status == "UNRESOLVED_TEST_MATRIX":
            reason = (
                "An official test exists, but its generated parameter matrix cannot "
                "yet be unfolded by the static specification extractor."
            )
        else:
            reason = (
                "No exact official manifest/test specification can currently be "
                "matched to this JIT root without guessing."
            )
        kernel["exclusion_reason"] = reason

    selected_spec_ids = {
        spec_id
        for root in qualified_roots.values()
        for spec_id in (root["original_spec_id"], *root["boundary_spec_ids"])
    }
    audit_specs = {
        item["spec_id"]: item
        for item in audit["specifications"]
        if item["spec_id"] in selected_spec_ids
    }
    if len(audit_specs) != 4 * len(qualified_roots):
        raise ValueError(
            f"{repository['project']} specification audit is not a complete 1+3 matrix"
        )

    recipes: dict[str, dict[str, Any]] = {}
    recipe_paths: dict[str, str] = {}
    recipe_root = corpus_root / "suite3" / "recipes" / f"{repository['source_id']}-auto"
    for spec_id in sorted(audit_specs):
        recipe = _tilelang_library_replay_recipe(audit_specs[spec_id])
        recipe_path = recipe_root / f"{safe_component(spec_id, maximum=180)}.json"
        atomic_write(recipe_path, pretty_json(recipe))
        recipes[spec_id] = recipe
        recipe_paths[spec_id] = recipe_path.relative_to(corpus_root).as_posix()

    for root_id in sorted(qualified_roots):
        audited_root = qualified_roots[root_id]
        candidate_id = audited_root["candidate_id"]
        kernel_id = kernel_by_candidate[candidate_id]
        revision_id = revision_by_candidate[candidate_id]
        kernel = kernels[kernel_id]
        original_id = audited_root["original_spec_id"]
        original_recipe = recipes[original_id]
        kernel["primary_original_spec_id"] = original_id
        for spec_id in (original_id, *audited_root["boundary_spec_ids"]):
            audited_spec = audit_specs[spec_id]
            recipe = recipes[spec_id]
            evidence_path = source_root / audited_spec["evidence_path"]
            actual_evidence_hash = _source_range_hash(
                evidence_path,
                audited_spec["evidence_start_line"],
                audited_spec["evidence_end_line"],
            )
            if actual_evidence_hash != audited_spec["evidence_source_sha256"]:
                raise ValueError(
                    f"{repository['project']} specification evidence drifted for {spec_id}"
                )
            evidence_id = (
                f"{repository['source_id']}-spec-evidence-"
                f"{stable_id(audited_spec['evidence_path'], audited_spec['evidence_function'])}"
            )
            evidence[evidence_id] = {
                "evidence_id": evidence_id,
                "kind": (
                    "UPSTREAM_TEST"
                    if audited_spec["evidence_kind"] == "UPSTREAM_TEST"
                    else "DOCUMENTATION"
                ),
                "repository_id": repository["source_id"],
                "commit": repository["commit"],
                "path": audited_spec["evidence_path"],
                "start_line": audited_spec["evidence_start_line"],
                "end_line": audited_spec["evidence_end_line"],
                "url": _permalink(
                    repository["repository_url"], repository["commit"],
                    audited_spec["evidence_path"], audited_spec["evidence_start_line"],
                    audited_spec["evidence_end_line"],
                ),
                "sha256": actual_evidence_hash,
                "rationale": (
                    f"Official {repository['project']} "
                    f"{audited_spec['evidence_kind'].lower()} supplying this source specification."
                ),
            }
            role = audited_spec["role"]
            delta = [] if role == "ORIGINAL" else _recipe_delta(original_recipe, recipe)
            if role == "BOUNDARY" and not delta:
                raise ValueError(f"TileLang boundary {spec_id} has no recipe delta")
            item = {
                "spec_id": spec_id,
                "kernel_id": kernel_id,
                "revision_id": revision_id,
                "role": role,
                "recipe": {
                    "path": recipe_paths[spec_id],
                    "recipe_id": recipe["recipe_id"],
                    "candidate_id": candidate_id,
                    "sha256": _hash(corpus_root / recipe_paths[spec_id]),
                },
                "delta": delta,
                "boundary_dimensions": audited_spec["boundary_dimensions"],
                "rule_ids": [],
                "evidence_ids": [f"source-{candidate_id}", evidence_id],
                "rationale": (
                    f"Source-derived {repository['project']} specialization; native "
                    "factory materialization and checker output remain unassessed."
                ),
                "execution": {
                    "selected_cta": [0, 0, 0],
                    "claim_scope": "SELECTED_CTA",
                    "symbol_policy": "ALL",
                    "symbols": [],
                    "engine": engine,
                },
                "expected_pipeline": {
                    "status": "UNASSESSED",
                    "allowed_diagnostic_codes": [],
                },
                "semantic_ground_truth": {
                    "class": "UNKNOWN",
                    "status": "UNADJUDICATED",
                    "method": "UNADJUDICATED",
                    "evidence_ids": [],
                    "rationale": (
                        "Upstream source supplies legal workload provenance; no checker "
                        "result is used as independent semantic ground truth."
                    ),
                },
            }
            if role == "BOUNDARY":
                item["parent_spec_id"] = audited_spec["parent_spec_id"]
                item["boundary_transition"] = (
                    "Official source-matrix variation across: "
                    + ", ".join(audited_spec["boundary_dimensions"])
                )
            specs[spec_id] = item


def seed_catalog(
    corpus_root: Path,
    catalog_path: Path,
    output_path: Path,
    source_roots: dict[str, Path],
) -> dict[str, Any]:
    lock_path = corpus_root / "suite3" / "lock.json"
    lock = _load(lock_path)
    repositories = {item["source_id"]: item for item in lock["repositories"]}
    toolchains = {item["frontend_source_id"]: item for item in lock["toolchains"]}
    catalog = _load(catalog_path)
    catalog["suite_lock_sha256"] = _hash(lock_path)

    evidence = {item["evidence_id"]: item for item in catalog["source_evidence"]}
    revisions = {item["revision_id"]: item for item in catalog["kernel_revisions"]}
    kernels = {item["kernel_id"]: item for item in catalog["kernels"]}
    specs = {item["spec_id"]: item for item in catalog["specifications"]}
    kernel_by_candidate = {
        revision["candidate_id"]: revision["kernel_id"]
        for revision in revisions.values()
    }
    revision_by_candidate = {
        revision["candidate_id"]: revision["revision_id"]
        for revision in revisions.values()
    }
    spec_by_recipe = {
        spec["recipe"]["recipe_id"]: spec["spec_id"]
        for spec in specs.values()
    }

    for repository_id in sorted(repositories):
        repository = repositories[repository_id]
        toolchain = toolchains.get(repository_id)
        dsl = repository["dsl"]
        source_root = source_roots[repository_id].resolve(strict=True)
        index_path = corpus_root / repository["source_index"]["path"]
        index = _load(index_path)
        for candidate in index["candidates"]:
            manifest = _load(corpus_root / candidate["manifest_path"])
            candidate_id = candidate["candidate_id"]
            kernel_id = kernel_by_candidate.get(candidate_id, _kernel_id(dsl, candidate))
            revision_id = revision_by_candidate.get(candidate_id, f"{kernel_id}@{repository['commit'][:8]}")

            source_evidence = next((
                item for item in evidence.values()
                if item["kind"] == "KERNEL_SOURCE"
                and item.get("repository_id") == repository_id
                and item.get("path") == manifest["upstream"]["relative_path"]
                and item.get("start_line") == manifest["upstream"]["start_line"]
                and item.get("end_line") == manifest["upstream"]["end_line"]
            ), None)
            if source_evidence is None:
                source_evidence = _source_evidence(manifest=manifest, repository=repository)
                evidence[source_evidence["evidence_id"]] = source_evidence

            if revision_id not in revisions:
                upstream = manifest["upstream"]
                revisions[revision_id] = {
                    "revision_id": revision_id,
                    "kernel_id": kernel_id,
                    "repository_id": repository_id,
                    "commit": repository["commit"],
                    "path": upstream["relative_path"],
                    "qualname": upstream["qualname"],
                    "candidate_id": candidate_id,
                    "git_blob": upstream["git_blob"],
                    "source_sha256": upstream["source_sha256"],
                    "selected_source_sha256": upstream["selected_source_sha256"],
                    "start_line": upstream["start_line"],
                    "end_line": upstream["end_line"],
                    "permalink": _permalink(
                        repository["repository_url"], repository["commit"], upstream["relative_path"],
                        upstream["start_line"], upstream["end_line"],
                    ),
                    "toolchain_id": None if toolchain is None else toolchain["toolchain_id"],
                }

            original_ids: list[str] = []
            for recipe_path_text in candidate["recipe_paths"]:
                recipe_path = corpus_root / recipe_path_text
                recipe = _load(recipe_path)
                if not recipe.get("lowerable"):
                    continue
                existing_spec_id = spec_by_recipe.get(recipe["recipe_id"])
                if existing_spec_id is not None:
                    original_ids.append(existing_spec_id)
                    continue
                spec_id = f"{kernel_id}-original-{stable_id(recipe['recipe_id'], length=10)}"
                launch_evidence = _launch_evidence(
                    recipe=recipe,
                    repository=repository,
                    source_root=source_root,
                )
                evidence_ids = [source_evidence["evidence_id"]]
                if launch_evidence is not None:
                    evidence[launch_evidence["evidence_id"]] = launch_evidence
                    evidence_ids.append(launch_evidence["evidence_id"])
                selected_cta = (recipe.get("compiler") or {}).get("selected_cta", [0, 0, 0])
                specs[spec_id] = {
                    "spec_id": spec_id,
                    "kernel_id": kernel_id,
                    "revision_id": revision_id,
                    "role": "ORIGINAL",
                    "recipe": {
                        "path": recipe_path_text,
                        "recipe_id": recipe["recipe_id"],
                        "candidate_id": candidate_id,
                        "sha256": _hash(recipe_path),
                    },
                    "delta": [],
                    "boundary_dimensions": [],
                    "rule_ids": [],
                    "evidence_ids": evidence_ids,
                    "rationale": f"Source-derived concrete upstream specialization. Recipe origin: {recipe['origin']}",
                    "execution": {
                        "selected_cta": selected_cta,
                        "claim_scope": _claim_scope(recipe),
                        "symbol_policy": "ALL",
                        "symbols": [],
                        "engine": lock["policy"]["default_checker_engine"],
                    },
                    "expected_pipeline": {
                        "status": "UNASSESSED",
                        "allowed_diagnostic_codes": [],
                    },
                    "semantic_ground_truth": {
                        "class": "UNKNOWN",
                        "status": "UNADJUDICATED",
                        "method": "UNADJUDICATED",
                        "evidence_ids": [],
                        "rationale": "The source specialization is preserved before checker output is observed; semantic adjudication and boundary review remain pending.",
                    },
                }
                spec_by_recipe[recipe["recipe_id"]] = spec_id
                original_ids.append(spec_id)

            if kernel_id not in kernels:
                kernel = {
                    "kernel_id": kernel_id,
                    "name": candidate["qualname"],
                    "project": repository["project"],
                    "dsl": dsl,
                    "candidate_kind": candidate["kind"],
                    "source_role": candidate["role"],
                    "admission": "CANDIDATE",
                    "launch_root_rationale": (
                        "The locked source index exposes this device definition; launch-root classification is preserved separately from extraction success."
                    ),
                    "revision_ids": [revision_id],
                    "exclusion_reason": _candidate_reason(candidate),
                }
                if original_ids:
                    kernel["primary_original_spec_id"] = sorted(original_ids)[0]
                kernels[kernel_id] = kernel
            else:
                kernel = kernels[kernel_id]
                kernel.setdefault("dsl", dsl)
                kernel.setdefault("candidate_kind", candidate["kind"])
                kernel.setdefault("source_role", candidate["role"])

    flaggems = repositories.get("flaggems-34bd6d68")
    if flaggems is not None:
        _merge_flaggems_specification_audit(
            corpus_root=corpus_root,
            source_root=source_roots[flaggems["source_id"]].resolve(strict=True),
            repository=flaggems,
            evidence=evidence,
            revisions=revisions,
            kernels=kernels,
            specs=specs,
            engine=lock["policy"]["default_checker_engine"],
        )

    for repository in repositories.values():
        for document in repository.get("audit_documents", []):
            if document["kind"] != "SPECIFICATION_AUDIT":
                continue
            audit_path = corpus_root / document["path"]
            audit = _load(audit_path)
            if audit.get("schema_version") != "pipeguard-tilelang-library-specification-audit-v1":
                continue
            _merge_tilelang_library_specification_audit(
                corpus_root=corpus_root,
                source_root=source_roots[repository["source_id"]].resolve(strict=True),
                repository=repository,
                audit_path=audit_path,
                evidence=evidence,
                revisions=revisions,
                kernels=kernels,
                specs=specs,
                engine=lock["policy"]["default_checker_engine"],
            )

    reviewed_path = corpus_root / "suite3" / "reviewed-specs.json"
    if reviewed_path.is_file():
        reviewed = _load(reviewed_path)
        if set(reviewed) != {
            "schema_version", "source_evidence", "kernel_overrides", "specifications"
        } or reviewed["schema_version"] != "pipeguard-upstream-reviewed-specs-v1":
            raise ValueError(f"invalid reviewed specification overlay: {reviewed_path}")
        for item in reviewed["source_evidence"]:
            evidence_id = item["evidence_id"]
            if evidence_id in evidence and evidence[evidence_id] != item:
                raise ValueError(f"reviewed evidence conflicts with generated record: {evidence_id}")
            evidence[evidence_id] = item
        for item in reviewed["specifications"]:
            spec_id = item["spec_id"]
            if spec_id in specs and specs[spec_id] != item:
                raise ValueError(f"reviewed specification conflicts with catalog: {spec_id}")
            specs[spec_id] = item
        allowed_override_fields = {
            "primary_original_spec_id", "launch_root_rationale", "exclusion_reason"
        }
        for override in reviewed["kernel_overrides"]:
            kernel_id = override["kernel_id"]
            if kernel_id not in kernels:
                raise ValueError(f"reviewed override references unknown kernel: {kernel_id}")
            unexpected = set(override) - allowed_override_fields - {"kernel_id"}
            if unexpected:
                raise ValueError(f"unsupported reviewed kernel override fields: {sorted(unexpected)}")
            kernels[kernel_id].update({
                key: value for key, value in override.items() if key != "kernel_id"
            })

    catalog["source_evidence"] = sorted(evidence.values(), key=lambda item: item["evidence_id"])
    catalog["kernel_revisions"] = sorted(revisions.values(), key=lambda item: item["revision_id"])
    catalog["kernels"] = sorted(kernels.values(), key=lambda item: item["kernel_id"])
    catalog["specifications"] = sorted(specs.values(), key=lambda item: item["spec_id"])
    atomic_write(output_path, pretty_json(catalog))
    validate_suite(corpus_root, catalog_path=output_path)
    return catalog


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Populate Suite 3 from every lowerable locked upstream recipe")
    parser.add_argument("--corpus-root", type=Path, default=Path(__file__).parents[1] / "corpus" / "upstream")
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tilelang-root", type=Path, required=True)
    parser.add_argument("--triton-root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        action="append",
        default=[],
        metavar="SOURCE_ID=PATH",
        help="additional locked repository root, repeatable",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    corpus_root = arguments.corpus_root.resolve(strict=True)
    catalog_path = (arguments.catalog or corpus_root / "suite3" / "catalog.json").resolve(strict=True)
    source_roots = {
        "tilelang-2c31e071": arguments.tilelang_root,
        "triton-61c02bcd": arguments.triton_root,
    }
    for assignment in arguments.source_root:
        source_id, separator, path = assignment.partition("=")
        if not separator or not source_id or not path:
            raise SystemExit(f"invalid --source-root assignment: {assignment}")
        if source_id in source_roots:
            raise SystemExit(f"duplicate source root for {source_id}")
        source_roots[source_id] = Path(path)
    catalog = seed_catalog(
        corpus_root,
        catalog_path,
        arguments.output.resolve(),
        source_roots,
    )
    print(json.dumps({
        "kernels": len(catalog["kernels"]),
        "specifications": len(catalog["specifications"]),
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
