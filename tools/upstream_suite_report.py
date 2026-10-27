#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from upstream_common import atomic_write, canonical_json, pretty_json, sha256_bytes, sha256_text
from upstream_suite import validate_checker_result, validate_suite


def _load(path: Path) -> Any:
    return json.loads(path.read_text())


def _hash(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def _checker_results(corpus_root: Path, check_root: Path) -> dict[str, list[dict[str, Any]]]:
    indexed: dict[str, list[dict[str, Any]]] = {}
    for path in sorted(check_root.rglob("*.result.json")):
        result = _load(path)
        validate_checker_result(corpus_root, result)
        indexed.setdefault(result["spec_id"], []).append(result)
    for path in sorted(check_root.rglob("capture-not-available.json")):
        result = _load(path)
        validate_checker_result(corpus_root, result)
        indexed.setdefault(result["spec_id"], []).append(result)
    return indexed


def _actual_verdicts(results: list[dict[str, Any]]) -> list[str]:
    return sorted(
        result["actual"]["verdict"]
        for result in results
        if result["status"] == "CHECKED"
    )


def _expectation_matches(spec: dict[str, Any], results: list[dict[str, Any]]) -> bool:
    if not results:
        return False
    expected_pipeline = spec["expected_pipeline"]["status"]
    if any(result["capture"]["status"] != expected_pipeline for result in results):
        return False
    expected_checker = spec.get("expected_checker")
    if expected_checker is None:
        return all(result["status"] == "CAPTURE_NOT_AVAILABLE" for result in results)
    if any(result["status"] != "CHECKED" for result in results):
        return False
    verdicts = _actual_verdicts(results)
    expected_verdict = expected_checker["verdict"]
    if expected_verdict == "SAFE":
        verdict_matches = bool(verdicts) and all(verdict == "SAFE" for verdict in verdicts)
    else:
        verdict_matches = expected_verdict in verdicts and "INCONCLUSIVE" not in verdicts
    if not verdict_matches:
        return False
    allowed = set(expected_checker["allowed_diagnostic_codes"])
    if not allowed:
        return True
    diagnostics = {
        result["actual"]["primary_diagnostic"]
        for result in results
        if result["status"] == "CHECKED"
    }
    return bool(diagnostics & allowed)


def _semantic_agreement(spec: dict[str, Any], verdicts: list[str]) -> str:
    ground_truth = spec["semantic_ground_truth"]
    if ground_truth["status"] != "CONFIRMED":
        return "NOT_ADJUDICATED"
    if not verdicts:
        return "NOT_CHECKED"
    expected = {
        "CORRECT": "SAFE",
        "DATA_HAZARD": "DATA_HAZARD",
        "SYNC_HAZARD": "SYNC_HAZARD",
    }.get(ground_truth["class"])
    if expected is None:
        return "NOT_COMPARABLE"
    if expected == "SAFE":
        matches = all(verdict == "SAFE" for verdict in verdicts)
    else:
        matches = expected in verdicts and "INCONCLUSIVE" not in verdicts
    return "MATCH" if matches else "MISMATCH"


def _markdown(summary: dict[str, Any]) -> str:
    counts = summary["counts"]
    lines = [
        "# Real upstream kernel suite report",
        "",
        f"- Candidate kernels: {counts['candidate_kernels']}",
        f"- Admitted kernels: {counts['admitted_kernels']}",
        f"- Catalog specifications: {counts['catalog_specifications']}",
        f"- Selected specifications: {counts['selected_specifications']}",
        f"- Checked specifications: {counts['checked_specs']}",
        f"- Capture unavailable: {counts['capture_unavailable_specs']}",
        f"- Expectation matches: {counts['expectation_matches']}",
        f"- Expectation mismatches: {counts['expectation_mismatches']}",
        "",
        "`UNSUPPORTED`, lowering failures, timeouts, and checker errors are excluded from SAFE counts.",
        "",
        "| Specification | Role | Pipeline | Checker | Scope | Expectation | Ground truth |",
        "|---|---|---|---|---|---|---|",
    ]
    for record in summary["results"]:
        checker = ", ".join(record["checker_verdicts"]) or "not checked"
        lines.append(
            f"| {record['spec_id']} | {record['role']} | {record['pipeline_status']} | "
            f"{checker} | {record['claim_scope']} | "
            f"{'match' if record['expectation_match'] else 'mismatch'} | "
            f"{record['semantic_agreement']} |"
        )
    return "\n".join(lines) + "\n"


def parse_arguments(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Join Suite 3 expectations with capture and checker results")
    parser.add_argument("--corpus-root", type=Path, default=Path(__file__).parents[1] / "corpus" / "upstream")
    parser.add_argument("--suite-lock", type=Path)
    parser.add_argument("--catalog", type=Path)
    parser.add_argument("--check-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--spec-id", action="append", default=[])
    parser.add_argument("--markdown", type=Path)
    arguments = parser.parse_args(argv)
    if len(arguments.spec_id) != len(set(arguments.spec_id)):
        parser.error("--spec-id values must be unique")
    return arguments


def main(argv: list[str] | None = None) -> int:
    arguments = parse_arguments(sys.argv[1:] if argv is None else argv)
    corpus_root = arguments.corpus_root.resolve(strict=True)
    check_root = arguments.check_root.resolve(strict=True)
    lock_path = None if arguments.suite_lock is None else arguments.suite_lock.resolve(strict=True)
    catalog_path = None if arguments.catalog is None else arguments.catalog.resolve(strict=True)
    lock, catalog, catalog_summary = validate_suite(corpus_root, lock_path=lock_path, catalog_path=catalog_path)
    lock_path = lock_path or corpus_root / "suite3" / "lock.json"
    catalog_path = catalog_path or corpus_root / "suite3" / "catalog.json"
    checked = _checker_results(corpus_root, check_root)
    all_specs = {item["spec_id"]: item for item in catalog["specifications"]}
    selected_ids = set(arguments.spec_id) if arguments.spec_id else set(all_specs)
    missing = selected_ids - set(all_specs)
    if missing:
        raise ValueError(f"unknown suite specification IDs: {sorted(missing)}")
    selected_specs = [all_specs[spec_id] for spec_id in sorted(selected_ids)]

    records: list[dict[str, Any]] = []
    for spec in selected_specs:
        results = checked.get(spec["spec_id"], [])
        pipeline_statuses = sorted({result["capture"]["status"] for result in results})
        pipeline_status = ",".join(pipeline_statuses) if pipeline_statuses else "MISSING"
        verdicts = _actual_verdicts(results)
        capture_diagnostic_codes = sorted({
            code
            for result in results
            for diagnostic in result.get("diagnostics", [])
            for code in diagnostic.get("details", {}).get("capture_diagnostic_codes", [])
        })
        records.append({
            "spec_id": spec["spec_id"],
            "kernel_id": spec["kernel_id"],
            "role": spec["role"],
            "claim_scope": spec["execution"]["claim_scope"],
            "pipeline_status": pipeline_status,
            "capture_diagnostic_codes": capture_diagnostic_codes,
            "checker_statuses": sorted({result["status"] for result in results}),
            "checker_verdicts": verdicts,
            "expectation_match": _expectation_matches(spec, results),
            "semantic_ground_truth": spec["semantic_ground_truth"]["class"],
            "semantic_adjudication": spec["semantic_ground_truth"]["status"],
            "semantic_agreement": _semantic_agreement(spec, verdicts),
            "result_digests": sorted(
                sha256_text(canonical_json(result))
                for result in results
            ),
        })

    status_counts = Counter(
        status
        for results in checked.values()
        for result in results
        for status in [result["status"]]
    )
    verdict_counts = Counter(
        verdict
        for results in checked.values()
        for verdict in _actual_verdicts(results)
    )
    summary = {
        "schema_version": "pipeguard-upstream-baseline-index-v1",
        "suite_id": lock["suite_id"],
        "suite_lock_sha256": _hash(lock_path),
        "catalog_sha256": _hash(catalog_path),
        "counts": {
            "candidate_kernels": catalog_summary.candidates,
            "admitted_kernels": catalog_summary.admitted_kernels,
            "catalog_specifications": len(catalog["specifications"]),
            "selected_specifications": len(selected_specs),
            "checked_specs": sum(any(item["status"] == "CHECKED" for item in checked.get(spec["spec_id"], [])) for spec in selected_specs),
            "capture_unavailable_specs": sum(any(item["status"] == "CAPTURE_NOT_AVAILABLE" for item in checked.get(spec["spec_id"], [])) for spec in selected_specs),
            "expectation_matches": sum(record["expectation_match"] for record in records),
            "expectation_mismatches": sum(not record["expectation_match"] for record in records),
        },
        "checker_statuses": dict(sorted(status_counts.items())),
        "checker_verdicts": dict(sorted(verdict_counts.items())),
        "results": records,
    }
    atomic_write(arguments.output.resolve(), pretty_json(summary))
    if arguments.markdown is not None:
        atomic_write(arguments.markdown.resolve(), _markdown(summary))
    print(canonical_json(summary["counts"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
