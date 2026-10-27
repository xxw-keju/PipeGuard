from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path.insert(0, str(TESTS))

from suite.frontend_pairs.catalog import PAIRS  # noqa: E402

SCHEMA_VERSION = "pipeguard-suite2-labeled-records-v1"
STATUS = "PRELIMINARY_BASELINES_MISSING_NOT_FOR_FINAL_FIGURE"
MIN_UNIQUE_KERNELS = 15
EXPECTED_INPUTS = {
    ".artifacts/frontend-pairs/paired-checks.json": "a3d547a0603897e10974f773d87b359bbab9267f2f017bf511c910def4161bf0",
    ".artifacts/frontend-pairs/tilelang-checks.json": "99c5da7c2f5442acfa4fb3d923ce333efc729bf91f205c5a8fca2f872421ac3f",
    ".artifacts/frontend-pairs/tilelang-results.json": "82b30c190eaa99145408b51d57b90118d1a42318506ed66052395d1de75c1769",
    ".artifacts/frontend-pairs/triton-checks.json": "af3cc960803f39fe346b4693e70a5efb135a975ba319dacf0f22bdd17780ddd2",
    ".artifacts/frontend-pairs/triton-results.json": "cb9a25a3c5c7468036b816021c10df39f3d735b4d628c4e4094a3e26d59f301e",
    "tests/suite/frontend_pairs/catalog.py": "e1e3514f555f78e61f051ed0f62c5f1a6656cdb5409dcdb6c61a3571625c0e3c",
    "tests/suite/pgir_direct/cases/catalog.py": "8d53f55d768aef42cc775046a1bcb4f85d5727107caa7e52178939ff98e3e04f",
}
PRIMITIVE_BY_FAMILY = {
    "CP-ASYNC": "cp.async",
    "CP-OVERLAP": "cp.async",
    "CP-WAIT": "cp.async",
    "CTA-BARRIER": "Barrier",
    "CTA-HANDOFF": "Barrier",
    "MB-REUSE": "mbarrier",
    "MBARRIER": "mbarrier",
    "MEM-RAW": "Memory",
    "MEM-WAW": "Memory",
    "MEMORY": "Memory",
    "TMA-LOAD": "TMA",
    "TMA-OBSERVE": "TMA",
    "TMA-PROXY": "Proxy",
    "TMA-STORE": "TMA",
    "TMA-STORE-ENDPOINT": "TMA",
    "WGMMA": "WGMMA",
    "WGMMA-ENDPOINT": "WGMMA",
}
DSLS = ("tilelang", "triton")
RESULT_CLASSES = ("SAFE", "DATA_HAZARD", "SYNC_HAZARD")
PRIMITIVES = ("Barrier", "mbarrier", "TMA", "cp.async", "WGMMA", "Proxy", "Memory")


class Suite2RecordsError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_bytes(value))


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": _relative(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def _verify_frozen_inputs() -> dict[str, dict[str, Any]]:
    refs = {}
    for relative, expected in EXPECTED_INPUTS.items():
        path = ROOT / relative
        if not path.is_file():
            raise Suite2RecordsError(f"missing frozen input: {relative}")
        ref = file_ref(path)
        if ref["sha256"] != expected:
            raise Suite2RecordsError(
                f"frozen input drift: {relative}: expected {expected}, got {ref['sha256']}"
            )
        refs[relative] = ref
    return refs


def _load_records(path: Path, expected_labels: set[str]) -> dict[str, dict[str, Any]]:
    value = json.loads(path.read_text())
    if not isinstance(value, list):
        raise Suite2RecordsError(f"expected list input: {_relative(path)}")
    records = {}
    for record in value:
        label = record.get("label")
        if not isinstance(label, str) or label in records:
            raise Suite2RecordsError(f"invalid or duplicate label in {_relative(path)}: {label!r}")
        records[label] = record
    if set(records) != expected_labels:
        missing = sorted(expected_labels - set(records))
        extra = sorted(set(records) - expected_labels)
        raise Suite2RecordsError(
            f"label inventory mismatch in {_relative(path)}: missing={missing}, extra={extra}"
        )
    return records


def _optional_file_ref(artifacts: Path, relative: str | None) -> dict[str, Any] | None:
    if relative is None:
        return None
    path = artifacts / relative
    if not path.is_file():
        return {"path": _relative(path), "status": "MISSING", "bytes": None, "sha256": None}
    return file_ref(path)


def _nvidia_baselines() -> dict[str, dict[str, Any]]:
    return {
        mode: {
            "status": "NOT_RUN_NO_EXECUTABLE_BINARY",
            "denominator_eligible": False,
            "diagnostic": None,
            "interpretation": (
                "compile-only Suite 2 artifacts contain no exact executable launch; "
                "absence of a run is neither SAFE, detection, nor miss"
            ),
        }
        for mode in ("synccheck", "memcheck")
    }


def _triton_baseline(dsl: str, capture: dict[str, Any]) -> dict[str, Any]:
    if dsl != "triton":
        return {
            "status": "OUT_OF_DOMAIN_OR_NOT_APPLICABLE",
            "denominator_eligible": False,
            "structural_acceptance_only": True,
            "interpretation": "no TileLang-native Triton checker is selected",
        }
    accepted = capture["status"] == "CAPTURED"
    return {
        "status": (
            "STRUCTURAL_COMPILER_ACCEPTED"
            if accepted
            else "STRUCTURAL_COMPILER_UNSUPPORTED"
        ),
        "denominator_eligible": False,
        "structural_acceptance_only": True,
        "interpretation": (
            "compiler/frontend structural acceptance is not a race-freedom verdict"
        ),
    }


def _git_identity() -> dict[str, Any]:
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "HEAD"], cwd=ROOT
    )
    return {
        "head": head,
        "tracked_binary_diff_sha256": sha256_bytes(tracked_diff),
        "direct_core": file_ref(ROOT / "src/pipeguard/direct_parametric.py"),
        "scope": (
            "builder-time consumer identity only; original Suite 2 input artifacts "
            "do not record their producing HEAD/core identity"
        ),
    }


def build_records(artifacts: Path) -> dict[str, Any]:
    artifacts = artifacts.resolve()
    if artifacts != (ROOT / ".artifacts/frontend-pairs").resolve():
        raise Suite2RecordsError(
            "this frozen v1 builder accepts only .artifacts/frontend-pairs"
        )
    input_refs = _verify_frozen_inputs()
    expected_labels = {case.label for case in PAIRS}
    if len(PAIRS) != 290 or len(expected_labels) != 290:
        raise Suite2RecordsError("Suite 2 catalog must contain 290 unique labels")

    captures = {
        dsl: _load_records(artifacts / f"{dsl}-results.json", expected_labels)
        for dsl in DSLS
    }
    checks = {
        dsl: _load_records(artifacts / f"{dsl}-checks.json", expected_labels)
        for dsl in DSLS
    }
    paired = json.loads((artifacts / "paired-checks.json").read_text())
    if len(paired) != 151 or len({record["label"] for record in paired}) != 151:
        raise Suite2RecordsError("paired-checks inventory must contain 151 unique labels")

    records = []
    for dsl in DSLS:
        for case in PAIRS:
            capture = captures[dsl][case.label]
            check = checks[dsl][case.label]
            for key, expected in (
                ("pair_id", case.pair_id),
                ("rule_id", case.rule_id),
                ("polarity", case.polarity),
            ):
                if capture.get(key) != expected or check.get(key) != expected:
                    raise Suite2RecordsError(
                        f"catalog mismatch for {dsl}/{case.label}/{key}"
                    )
            if (
                capture.get("expected_stage", case.expected_stage) != case.expected_stage
                or check.get("expected_stage") != case.expected_stage
            ):
                raise Suite2RecordsError(
                    f"catalog mismatch for {dsl}/{case.label}/expected_stage"
                )
            if check.get("expected_verdict") != case.expected_verdict:
                raise Suite2RecordsError(
                    f"oracle verdict mismatch for {dsl}/{case.label}"
                )
            if check.get("expected_diagnostic") != case.expected_diagnostic:
                raise Suite2RecordsError(
                    f"oracle diagnostic mismatch for {dsl}/{case.label}"
                )
            if check.get("capture_status") != capture["status"]:
                raise Suite2RecordsError(
                    f"capture/check status mismatch for {dsl}/{case.label}"
                )

            checked = check["status"] == "CHECKED"
            strict_agreement = None
            if checked and case.expected_stage == "CHECKER":
                strict_agreement = (
                    check.get("actual_verdict") == case.expected_verdict
                    and (
                        case.expected_diagnostic is None
                        or check.get("actual_diagnostic") == case.expected_diagnostic
                    )
                )
            stored_agreement = check.get("matches_expected")
            expected_stored = bool(strict_agreement) if case.expected_stage == "CHECKER" else (
                check["status"] == "MODEL_REJECTED"
            )
            if stored_agreement is not None and stored_agreement != expected_stored:
                raise Suite2RecordsError(
                    f"stored agreement mismatch for {dsl}/{case.label}"
                )

            raw_ref = _optional_file_ref(artifacts, capture.get("raw_path"))
            if raw_ref is None:
                raw_ref = {"path": None, "status": "MISSING", "bytes": None, "sha256": None}
            snapshot_ref = _optional_file_ref(artifacts, capture.get("snapshot_path"))
            pgir_ref = _optional_file_ref(artifacts, check.get("pgir_path"))
            records.append({
                "record_id": f"suite2:{dsl}:{case.label}",
                "suite": "suite2",
                "dsl": dsl,
                "label": case.label,
                "pair_id": case.pair_id,
                "rule_id": case.rule_id,
                "rule_ids": list(case.rule_ids),
                "family": case.family,
                "primitive": PRIMITIVE_BY_FAMILY[case.family],
                "scenario_id": case.scenario_id,
                "variant": case.variant,
                "polarity": case.polarity,
                "shape": case.shape,
                "oracle": {
                    "stage": case.expected_stage,
                    "result_class": case.expected_verdict,
                    "expected_verdict": case.expected_verdict,
                    "expected_diagnostic": case.expected_diagnostic,
                    "possible_bad_execution": list(case.possible_bad_execution),
                    "provenance": {
                        "kind": "DIRECT_PGIR_REFERENCE_CATALOG",
                        "independent_native_execution": False,
                        "catalogs": [
                            input_refs["tests/suite/pgir_direct/cases/catalog.py"],
                            input_refs["tests/suite/frontend_pairs/catalog.py"],
                        ],
                    },
                },
                "capture": {
                    "status": capture["status"],
                    "diagnostic": capture.get("diagnostic"),
                    "operations": capture.get("operations"),
                    "raw": raw_ref,
                    "snapshot": snapshot_ref,
                },
                "pipeguard": {
                    "status": check["status"],
                    "actual_verdict": check.get("actual_verdict"),
                    "actual_diagnostic": check.get("actual_diagnostic"),
                    "diagnostic": check.get("diagnostic"),
                    "strict_oracle_agreement": strict_agreement,
                    "denominator_eligible": case.expected_stage == "CHECKER",
                    "events": check.get("events"),
                    "n_ext": check.get("n_ext"),
                    "t_eff": check.get("t_eff"),
                    "pgir": pgir_ref,
                },
                "baselines": {
                    "nvidia": _nvidia_baselines(),
                    "triton_compiler_verifier": _triton_baseline(dsl, capture),
                },
                "kernel_identity": {
                    "method": "RAW_SOURCE_CONTENT_SHA256",
                    "sha256": raw_ref["sha256"],
                },
                "final_figure_eligible": False,
            })

    records.sort(key=lambda item: (item["dsl"], item["label"]))
    if len(records) != 580 or len({item["record_id"] for item in records}) != 580:
        raise Suite2RecordsError("expected exactly 580 unique DSL/label records")

    cells = []
    for dsl in DSLS:
        for result_class in RESULT_CLASSES:
            for primitive in PRIMITIVES:
                selected = [
                    record for record in records
                    if record["dsl"] == dsl
                    and record["oracle"]["stage"] == "CHECKER"
                    and record["oracle"]["result_class"] == result_class
                    and record["primitive"] == primitive
                ]
                checked = [r for r in selected if r["pipeguard"]["status"] == "CHECKED"]
                matching = [r for r in checked if r["pipeguard"]["strict_oracle_agreement"] is True]
                mismatch = [r for r in checked if r["pipeguard"]["strict_oracle_agreement"] is False]
                unique_raw = {
                    r["kernel_identity"]["sha256"]
                    for r in selected
                    if r["kernel_identity"]["sha256"] is not None
                }
                checked_unique = {
                    r["kernel_identity"]["sha256"]
                    for r in checked
                    if r["kernel_identity"]["sha256"] is not None
                }
                blockers = []
                if len(unique_raw) < MIN_UNIQUE_KERNELS:
                    blockers.append("INSUFFICIENT_CONTENT_UNIQUE_KERNELS")
                if len(checked_unique) < MIN_UNIQUE_KERNELS:
                    blockers.append("INSUFFICIENT_CHECKED_UNIQUE_KERNELS")
                if len(checked) != len(selected):
                    blockers.append("PIPEGUARD_ROWS_INCOMPLETE")
                if mismatch:
                    blockers.append("PIPEGUARD_ORACLE_MISMATCH")
                blockers.append("BASELINES_NOT_RUN_NO_EXECUTABLE_BINARY")
                cells.append({
                    "dsl": dsl,
                    "result_class": result_class,
                    "primitive": primitive,
                    "catalog_labels": len(selected),
                    "content_unique_kernels": len(unique_raw),
                    "pipeguard_checked_labels": len(checked),
                    "pipeguard_checked_unique_kernels": len(checked_unique),
                    "pipeguard_oracle_matches": len(matching),
                    "pipeguard_oracle_mismatches": len(mismatch),
                    "baseline_complete_unique_kernels": 0,
                    "minimum_unique_kernels": MIN_UNIQUE_KERNELS,
                    "ready": False,
                    "blockers": blockers,
                })

    checker_records = [r for r in records if r["oracle"]["stage"] == "CHECKER"]
    checked_checker = [r for r in checker_records if r["pipeguard"]["status"] == "CHECKED"]
    matching_checker = [
        r for r in checked_checker if r["pipeguard"]["strict_oracle_agreement"] is True
    ]
    mismatch_records = [
        r for r in checked_checker if r["pipeguard"]["strict_oracle_agreement"] is False
    ]
    result_counts = {
        dsl: {
            result_class: sum(
                r["dsl"] == dsl
                and r["oracle"]["stage"] == "CHECKER"
                and r["oracle"]["result_class"] == result_class
                for r in records
            )
            for result_class in RESULT_CLASSES
        }
        for dsl in DSLS
    }
    status_counts = {
        dsl: dict(sorted(Counter(
            r["pipeguard"]["status"] for r in records if r["dsl"] == dsl
        ).items()))
        for dsl in DSLS
    }
    blockers = [
        "BASELINES_NOT_RUN_NO_EXECUTABLE_BINARY",
        f"PIPEGUARD_INCOMPLETE_CHECKER_ROWS_{len(checker_records) - len(checked_checker)}",
        f"PIPEGUARD_ORACLE_MISMATCHES_{len(mismatch_records)}",
        "CELL_MINIMUM_15_CONTENT_UNIQUE_KERNELS_NOT_MET",
        "ORIGINAL_PRODUCER_IDENTITY_ABSENT_FROM_SOURCE_ARTIFACTS",
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "evidence_level": "PRELIMINARY",
        "final_figure_eligible": False,
        "all_correct": False,
        "denominator_policy": {
            "unit": "content-unique raw source kernel within DSL/result-class/primitive cell",
            "minimum_unique_kernels_per_displayed_cell": MIN_UNIQUE_KERNELS,
            "unsupported_timeout_na_are_safe": False,
            "mismatches_retained": True,
            "model_stage_rows_in_checker_cells": False,
        },
        "input_bindings": input_refs,
        "builder": file_ref(Path(__file__)),
        "current_consumer_identity": _git_identity(),
        "producer_provenance_limitation": (
            "The frozen source artifacts do not contain original producing HEAD, dirty diff, "
            "checker core, executable binary, or baseline-run identity. Current consumer "
            "identity is recorded separately and is not relabeled as historical producer identity."
        ),
        "summary": {
            "records": len(records),
            "checker_stage_records": len(checker_records),
            "model_stage_records": len(records) - len(checker_records),
            "pipeguard_checked_checker_records": len(checked_checker),
            "pipeguard_matching_checker_records": len(matching_checker),
            "pipeguard_mismatch_checker_records": len(mismatch_records),
            "pipeguard_incomplete_checker_records": len(checker_records) - len(checked_checker),
            "oracle_result_counts_by_dsl": result_counts,
            "pipeguard_status_counts_by_dsl": status_counts,
            "candidate_cells": len(cells),
            "ready_cells": sum(cell["ready"] for cell in cells),
            "baseline_complete_records": 0,
        },
        "mismatches": [
            {
                "record_id": record["record_id"],
                "dsl": record["dsl"],
                "label": record["label"],
                "expected_verdict": record["oracle"]["expected_verdict"],
                "expected_diagnostic": record["oracle"]["expected_diagnostic"],
                "actual_verdict": record["pipeguard"]["actual_verdict"],
                "actual_diagnostic": record["pipeguard"]["actual_diagnostic"],
            }
            for record in mismatch_records
        ],
        "candidate_cells": cells,
        "final_blockers": blockers,
        "records": records,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifacts", type=Path, default=ROOT / ".artifacts/frontend-pairs"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--require-final", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = build_records(args.artifacts)
        if args.require_final and not payload["final_figure_eligible"]:
            raise Suite2RecordsError(
                "final Suite 2 gate failed: " + ", ".join(payload["final_blockers"])
            )
        expected = canonical_bytes(payload)
        if args.check:
            if not args.output.is_file():
                raise Suite2RecordsError(f"output missing: {args.output}")
            actual = args.output.read_bytes()
            if actual != expected:
                raise Suite2RecordsError(
                    f"output mismatch: expected {sha256_bytes(expected)}, got {sha256_bytes(actual)}"
                )
        else:
            write_new(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            "records": payload["summary"]["records"],
            "checker_stage_records": payload["summary"]["checker_stage_records"],
            "checked": payload["summary"]["pipeguard_checked_checker_records"],
            "matching": payload["summary"]["pipeguard_matching_checker_records"],
            "mismatches": payload["summary"]["pipeguard_mismatch_checker_records"],
            "ready_cells": payload["summary"]["ready_cells"],
            "final_figure_eligible": payload["final_figure_eligible"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, Suite2RecordsError) as error:
        print(f"suite2 records error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
