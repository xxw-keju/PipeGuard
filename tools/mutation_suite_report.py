from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mutation_suite import load_json, sha256_file, validate_mutation_suite  # noqa: E402
from upstream_common import atomic_write, canonical_json, pretty_json  # noqa: E402


def _result_digest(path: Path) -> str:
    return sha256_file(path)


def build_index(repository_root: Path, run_roots: list[Path]) -> dict[str, Any]:
    lock, catalog, baseline, summary = validate_mutation_suite(repository_root)
    known = {item["case_id"] for item in catalog["cases"]}
    indexed: dict[str, list[tuple[Path, dict[str, Any]]]] = {}
    for run_root in run_roots:
        for path in sorted(run_root.rglob("result.json")):
            item = load_json(path)
            case_id = item.get("case_id")
            if case_id not in known:
                continue
            indexed.setdefault(case_id, []).append((path, item))
    results = []
    for case_id, records in sorted(indexed.items()):
        pipeline_values = [item.get("pipeline_status", "UNASSESSED") for _path, item in records]
        numeric_values = [item.get("numeric_status", "NOT_RUN") for _path, item in records]
        checker_values = [item.get("checker_status", "NOT_RUN") for _path, item in records]
        pipeline = next((value for value in reversed(pipeline_values) if value != "UNASSESSED"), "UNASSESSED")
        numeric = next((value for value in reversed(numeric_values) if value != "NOT_RUN"), "NOT_RUN")
        checker = next((value for value in reversed(checker_values) if value != "NOT_RUN"), "NOT_RUN")
        artifact_keys = {key for _path, item in records for key in item.get("artifacts", {})}
        lowering_status = "LOWERED" if any(key.startswith("raw:") for key in artifact_keys) else ("NOT_RUN" if pipeline == "UNASSESSED" else "FAILED")
        capture_status = "CAPTURED" if any(key.startswith("snapshot:") for key in artifact_keys) else ("NOT_RUN" if pipeline == "UNASSESSED" else "UNSUPPORTED")
        pgir_status = "PARSED" if any(key.startswith("pgir:") for key in artifact_keys) else ("NOT_RUN" if pipeline == "UNASSESSED" else "UNSUPPORTED")
        results.append({
            "case_id": case_id,
            "lowering_status": lowering_status,
            "capture_status": capture_status,
            "pgir_status": pgir_status,
            "pipeline_status": pipeline,
            "numeric_status": numeric,
            "checker_status": checker,
            "checker_verdicts": sorted({verdict for _path, item in records for verdict in item.get("checker_verdicts", [])}),
            "diagnostic_codes": sorted({entry["code"] for _path, item in records for entry in item.get("diagnostics", [])}),
            "result_digests": sorted({_result_digest(path) for path, _item in records}),
        })
    counts = {
        "kernels": summary.kernels,
        "original_variants": summary.original_variants,
        "mutant_variants": summary.mutant_variants,
        "original_workloads": summary.original_workloads,
        "boundary_workloads": summary.boundary_workloads,
        "cases": summary.cases,
        "assessed_cases": len(results),
        "lowered_cases": sum(item["lowering_status"] == "LOWERED" for item in results),
        "captured_snapshot_cases": sum(item["capture_status"] == "CAPTURED" for item in results),
        "parsed_pgir_cases": sum(item["pgir_status"] == "PARSED" for item in results),
        "checked_cases": sum(item["checker_status"] == "CHECKED" for item in results),
        "numeric_pass_cases": sum(item["numeric_status"] == "PASS" for item in results),
        "numeric_mismatch_cases": sum(item["numeric_status"] == "NUMERIC_MISMATCH" for item in results),
        "numeric_hang_cases": sum(item["numeric_status"] == "HANG" for item in results),
        "numeric_compile_error_cases": sum(item["numeric_status"] == "COMPILE_ERROR" for item in results),
    }
    return {
        "schema_version": "pipeguard-mutation-baseline-index-v1",
        "suite_id": lock["suite_id"],
        "suite_lock_sha256": sha256_file(repository_root / "corpus/mutations/suite4/lock.json"),
        "catalog_sha256": sha256_file(repository_root / "corpus/mutations/suite4/catalog.json"),
        "counts": counts,
        "results": results,
    }


def summary(index: dict[str, Any]) -> dict[str, Any]:
    return {
        "counts": index["counts"],
        "lowering_statuses": dict(sorted(Counter(item["lowering_status"] for item in index["results"]).items())),
        "capture_statuses": dict(sorted(Counter(item["capture_status"] for item in index["results"]).items())),
        "pgir_statuses": dict(sorted(Counter(item["pgir_status"] for item in index["results"]).items())),
        "pipeline_statuses": dict(sorted(Counter(item["pipeline_status"] for item in index["results"]).items())),
        "numeric_statuses": dict(sorted(Counter(item["numeric_status"] for item in index["results"]).items())),
        "checker_statuses": dict(sorted(Counter(item["checker_status"] for item in index["results"]).items())),
        "checker_verdicts": dict(sorted(Counter(verdict for item in index["results"] for verdict in item["checker_verdicts"]).items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a compact Suite 4 result index")
    parser.add_argument("--repository-root", type=Path, default=ROOT)
    parser.add_argument("--run-root", type=Path, action="append", required=True)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()
    repository_root = args.repository_root.resolve(strict=True)
    index = build_index(repository_root, [path.resolve(strict=True) for path in args.run_root])
    if args.write:
        atomic_write(repository_root / "corpus/mutations/suite4/baseline-index.json", pretty_json(index))
    print(canonical_json(summary(index)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
