#!/usr/bin/env python3
from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
CATALOG = ROOT / "corpus/mutations/suite4/catalog.json"
LOCK = ROOT / "corpus/mutations/suite4/lock.json"
BASELINE_INDEX = ROOT / "corpus/mutations/suite4/baseline-index.json"
PRELIMINARY = ROOT / ".artifacts/final-suite4-gate/preliminary-two-dsl-matrix-35df.json"
SHARED = ROOT / ".artifacts/final-suite4-gate/shared-racecheck"
REPLAY_ROOT = ROOT / ".artifacts/final-suite4-gate/final-candidate-v1"
SHARED_REPLAY = REPLAY_ROOT / "shared-current-core-replay.json"
CORE = ROOT / "src/pipeguard/direct_parametric.py"
CP_ASYNC_STATIC = ROOT / ".artifacts/final-checker-cpasync-exact-r/cp-async-consumer-gate.json"
CP_ASYNC_DYNAMIC = ROOT / ".artifacts/final-checker-cpasync-exact-r/cp-async-dynamic-simulation.json"

EXPECTED = {
    CATALOG: "127d7c6759d7da49dae1af2c65b9d48086ca5c34ac37609ce06c466388a89998",
    LOCK: "74e27e3b8e2d3de1e23b3b9b6bc51cfc9a4c85940f8b0164ec5cbcf20bee6f54",
    BASELINE_INDEX: "21a172daf61fc9de975e3677e7bed29d03f00287deca977c9856da3ec5bd796a",
    PRELIMINARY: "efba486d0a9ecf47c9985d134c9a8cab5c88454e43b6a397bfbe4fba3f402817",
    SHARED / "shared-race-binary-pair.json": "7a3e4e22c9d03f93c2f711256f15db55543d57a1f3d4ac6c698fba56c6e6d46c",
    SHARED / "racecheck-baseline.json": "3300683283e12447b3474dd2105d8f6f772170a5188ee89c359bc3037b7e72f9",
    SHARED / "remaining-sanitizer-baselines.json": "0d75b5daaee4c128836f1e80a37640722c7b1227ed3d29b5a80b20359fb87307",
    SHARED_REPLAY: "7ef33c1eb9a94e11636b73db06b115047005b2450993812be72c63fc060756fc",
    CORE: "35df5f3e83268a5711ff65c2894897506fe7389de437dc4325932aa51620d531",
    CP_ASYNC_STATIC: "15aa0e9358a1d27c0e44f37b33dbdb4ffdb59d33ddbd3cd58e1f3e6f36797fd9",
    CP_ASYNC_DYNAMIC: "7f39b709d6eeb21d379c82c1a3cf3b433aa1d7c309cdaf6b4cc492359307a554",
}


class FinalCandidateError(ValueError):
    pass


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise FinalCandidateError(f"JSON root must be an object: {path}")
    return value


def relative(path: Path) -> str:
    return str(path.relative_to(ROOT))


def file_ref(path: Path, kind: str) -> dict[str, Any]:
    if not path.is_file():
        raise FinalCandidateError(f"missing required {kind}: {path}")
    return {
        "path": relative(path),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
        "kind": kind,
    }


def verify_frozen_inputs() -> None:
    for path, expected in EXPECTED.items():
        actual = sha256_file(path)
        if actual != expected:
            raise FinalCandidateError(
                f"frozen input drift for {relative(path)}: expected {expected}, got {actual}"
            )


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def producer_identity(generator: Path) -> dict[str, Any]:
    return {
        "branch": git("branch", "--show-current").decode().strip(),
        "head": git("rev-parse", "HEAD").decode().strip(),
        "tracked_dirty_diff_sha256": hashlib.sha256(
            git("diff", "--binary", "HEAD")
        ).hexdigest(),
        "direct_core": file_ref(CORE, "CHECKER_CORE"),
        "generator": file_ref(generator, "GENERATOR"),
    }


def shared_files(generator: Path) -> list[dict[str, Any]]:
    names = {
        "shared-cross-lane-control.ttgir": "TTGIR_CONTROL_ANCESTOR",
        "shared-cross-lane-no-barrier.ptx": "HAZARD_FINAL_PTX",
        "shared-cross-lane-safe.ptx": "SAFE_FINAL_PTX",
        "shared-cross-lane-no-barrier.cubin": "HAZARD_EXACT_BINARY",
        "shared-cross-lane-safe.cubin": "SAFE_EXACT_BINARY",
        "shared-cross-lane-no-barrier.snapshot.json": "HAZARD_SNAPSHOT",
        "shared-cross-lane-safe.snapshot.json": "SAFE_SNAPSHOT",
        "shared-cross-lane-no-barrier.pgir.json": "HAZARD_PGIR",
        "shared-cross-lane-safe.pgir.json": "SAFE_PGIR",
        "shared-cross-lane-no-barrier.checker.json": "HISTORICAL_HAZARD_CHECKER",
        "shared-cross-lane-safe.checker.json": "HISTORICAL_SAFE_CHECKER",
        "shared-race-binary-pair.json": "PAIR_AUDIT",
        "shared-race-launcher.cu": "LAUNCHER_SOURCE",
        "shared-race-launcher": "LAUNCHER_BINARY",
        "racecheck-baseline.json": "RACECHECK_BASELINE",
        "remaining-sanitizer-baselines.json": "OTHER_SANITIZER_BASELINES",
    }
    for case in ("no-barrier", "barrier-control"):
        for mode in ("racecheck", "synccheck", "memcheck", "initcheck"):
            for stream in ("stdout", "stderr"):
                names[f"{case}.{mode}.{stream}"] = "BASELINE_RAW_OUTPUT"
    paths = [file_ref(SHARED / name, kind) for name, kind in sorted(names.items())]
    paths.extend([
        file_ref(REPLAY_ROOT / "shared-no-barrier-hazard.request.json", "CURRENT_CORE_REQUEST"),
        file_ref(REPLAY_ROOT / "shared-no-barrier-hazard.checker.json", "CURRENT_CORE_CHECKER"),
        file_ref(REPLAY_ROOT / "shared-barrier-control.request.json", "CURRENT_CORE_REQUEST"),
        file_ref(REPLAY_ROOT / "shared-barrier-control.checker.json", "CURRENT_CORE_CHECKER"),
        file_ref(SHARED_REPLAY, "CURRENT_CORE_REPLAY_RECEIPT"),
        file_ref(CATALOG, "PARENT_CATALOG"),
        file_ref(LOCK, "PARENT_LOCK"),
        file_ref(BASELINE_INDEX, "HISTORICAL_BASELINE_INDEX"),
        file_ref(PRELIMINARY, "PRELIMINARY_MATRIX"),
        file_ref(CP_ASYNC_STATIC, "OPTIONAL_CP_ASYNC_STATIC_GATE"),
        file_ref(CP_ASYNC_DYNAMIC, "OPTIONAL_CP_ASYNC_DYNAMIC_MODEL"),
        file_ref(CORE, "CHECKER_CORE"),
        file_ref(generator, "GENERATOR"),
    ])
    return sorted(paths, key=lambda item: item["path"])


def build_supplemental_lock(generator: Path) -> dict[str, Any]:
    verify_frozen_inputs()
    pair = load_json(SHARED / "shared-race-binary-pair.json")
    replay = load_json(SHARED_REPLAY)
    if replay.get("status") != "CURRENT_CORE_REPLAY_COMPLETE":
        raise FinalCandidateError("shared pair current-core replay is incomplete")
    replay_by_id = {record["case_id"]: record for record in replay["records"]}
    if replay["summary"] != {"rows": 2, "hazards": 1, "safe_controls": 1, "agreements": 2}:
        raise FinalCandidateError("shared pair replay summary mismatch")
    return {
        "schema_version": "pipeguard-suite4-supplemental-input-lock-v1",
        "suite_id": "suite4",
        "revision": 1,
        "status": "FROZEN",
        "parent_catalog": {
            "path": relative(CATALOG),
            "sha256": EXPECTED[CATALOG],
            "counts": {"kernels": 10, "variants": 94, "mutants": 84, "cases": 376},
        },
        "parent_lock": {"path": relative(LOCK), "sha256": EXPECTED[LOCK]},
        "baseline_index": {
            "path": relative(BASELINE_INDEX),
            "sha256": EXPECTED[BASELINE_INDEX],
            "result_count": 244,
            "role": "HISTORICAL_PARTIAL_COVERAGE_NOT_FINAL_ROW_INDEX",
        },
        "producer": producer_identity(generator),
        "denominator_policy": {
            "required_rows": 6,
            "pairs": 3,
            "positives": 3,
            "controls": 3,
            "catalog_rows": 4,
            "supplemental_rows": 2,
            "fixed_before_optional_cp_async": True,
        },
        "inputs": [
            {
                "input_id": "shared-no-barrier-hazard",
                "pair_id": "shared-exact-binary",
                "stratum": "SUPPLEMENTAL_FINAL_PTX_MUTATION",
                "evaluation_role": "POSITIVE",
                "catalog_membership": False,
                "expected_verdict": "DATA_HAZARD",
                "actual_verdict": replay_by_id["shared-no-barrier-hazard"]["actual_verdict"],
                "exact_binary": file_ref(SHARED / "shared-cross-lane-no-barrier.cubin", "HAZARD_EXACT_BINARY"),
                "pgir": file_ref(SHARED / "shared-cross-lane-no-barrier.pgir.json", "HAZARD_PGIR"),
                "checker": file_ref(REPLAY_ROOT / "shared-no-barrier-hazard.checker.json", "CURRENT_CORE_CHECKER"),
                "audit": file_ref(SHARED / "shared-race-binary-pair.json", "PAIR_AUDIT"),
                "baseline": file_ref(SHARED / "racecheck-baseline.json", "RACECHECK_BASELINE"),
                "launch": pair["launch"],
            },
            {
                "input_id": "shared-barrier-control",
                "pair_id": "shared-exact-binary",
                "stratum": "SUPPLEMENTAL_FINAL_PTX_CONTROL",
                "evaluation_role": "SAFE_CONTROL",
                "catalog_membership": False,
                "expected_verdict": "SAFE",
                "actual_verdict": replay_by_id["shared-barrier-control"]["actual_verdict"],
                "exact_binary": file_ref(SHARED / "shared-cross-lane-safe.cubin", "SAFE_EXACT_BINARY"),
                "pgir": file_ref(SHARED / "shared-cross-lane-safe.pgir.json", "SAFE_PGIR"),
                "checker": file_ref(REPLAY_ROOT / "shared-barrier-control.checker.json", "CURRENT_CORE_CHECKER"),
                "audit": file_ref(SHARED / "shared-race-binary-pair.json", "PAIR_AUDIT"),
                "baseline": file_ref(SHARED / "racecheck-baseline.json", "RACECHECK_BASELINE"),
                "launch": pair["launch"],
            },
        ],
        "optional_inputs": [
            {
                "input_id": "cp-async-executable-pair",
                "denominator": False,
                "status": "PENDING_UNSUPPORTED",
                "reason": "static owner-local and dynamic-model evidence exists, but no exact executable positive/control pair is frozen",
                "static_gate": file_ref(CP_ASYNC_STATIC, "OPTIONAL_CP_ASYNC_STATIC_GATE"),
                "dynamic_model": file_ref(CP_ASYNC_DYNAMIC, "OPTIONAL_CP_ASYNC_DYNAMIC_MODEL"),
            }
        ],
        "files": shared_files(generator),
    }


def _catalog_records(preliminary: dict[str, Any]) -> list[dict[str, Any]]:
    records = deepcopy(preliminary["records"])
    if len(records) != 4:
        raise FinalCandidateError("preliminary matrix must contain four catalog rows")
    for record in records:
        mutation = record.get("mutation_rule_id")
        record["catalog_membership"] = True
        record["catalog_role"] = "ORIGINAL" if mutation is None else "MUTANT"
        record["evaluation_role"] = (
            "POSITIVE" if record["audit"]["label"] == "DATA_HAZARD" else "SAFE_CONTROL"
        )
        record["stratum"] = "CATALOG_SOURCE_MUTATION"
        record["pair_id"] = (
            "tilelang-m6-original" if record["dsl"] == "tilelang" else "triton-m14-m15"
        )
        record["final_figure_eligible"] = True
        if record["dsl"] == "tilelang":
            racecheck = record["baselines"]["nvidia"]["racecheck"]
            racecheck["binary_identity"] = "NON_EXACT_BINARY_APPLICABILITY_ONLY"
            racecheck["final_matrix_classification"] = "OUT_OF_DOMAIN_OR_NOT_APPLICABLE"
            racecheck["denominator_eligible"] = False
            racecheck["integrity_note"] = (
                "The baseline-era cubin differs from the current-core PipeGuard replay; "
                "TMA bulk-copy hazard coverage is not explicitly documented, so silence is not a miss."
            )
        else:
            record["baselines"]["nvidia"]["initcheck"]["denominator_eligible"] = False
            record["baselines"]["nvidia"]["initcheck"]["integrity_note"] = (
                "Contaminated timeout is retained and excluded from correctness statistics."
            )
    return records


def _shared_records(lock_document: dict[str, Any]) -> list[dict[str, Any]]:
    pair = load_json(SHARED / "shared-race-binary-pair.json")
    racecheck = load_json(SHARED / "racecheck-baseline.json")
    remaining = load_json(SHARED / "remaining-sanitizer-baselines.json")
    replay = load_json(SHARED_REPLAY)
    replay_by_id = {record["case_id"]: record for record in replay["records"]}
    race_by_case = {record["case"]: record for record in racecheck["records"]}
    other_by_case: dict[str, dict[str, Any]] = {}
    for record in remaining["records"]:
        other_by_case.setdefault(record["case"], {})[record["mode"]] = record
    rows = []
    for case_name, case_id, role, label, ptx_name, cubin_name, snapshot_name, pgir_name in (
        (
            "no-barrier",
            "shared-no-barrier-hazard",
            "POSITIVE",
            "DATA_HAZARD",
            "shared-cross-lane-no-barrier.ptx",
            "shared-cross-lane-no-barrier.cubin",
            "shared-cross-lane-no-barrier.snapshot.json",
            "shared-cross-lane-no-barrier.pgir.json",
        ),
        (
            "barrier-control",
            "shared-barrier-control",
            "SAFE_CONTROL",
            "CORRECT",
            "shared-cross-lane-safe.ptx",
            "shared-cross-lane-safe.cubin",
            "shared-cross-lane-safe.snapshot.json",
            "shared-cross-lane-safe.pgir.json",
        ),
    ):
        checked = replay_by_id[case_id]
        checker_path = Path(checked["checker"]["path"])
        pipeguard = load_json(checker_path)
        rows.append({
            "suite": "suite4",
            "case_id": case_id,
            "dsl": "triton",
            "project": "Triton",
            "kernel": "shared-cross-lane",
            "pair_id": "shared-exact-binary",
            "stratum": "SUPPLEMENTAL_FINAL_PTX_MUTATION",
            "catalog_membership": False,
            "catalog_role": "SUPPLEMENTAL",
            "evaluation_role": role,
            "mutation_family": "SHARED_MEMORY_ORDER",
            "mutation_rule_id": "SUPPLEMENTAL-FINAL-PTX-REMOVE-BARRIER" if role == "POSITIVE" else None,
            "selected_cta": [0, 0, 0],
            "source": file_ref(SHARED / ptx_name, "FINAL_PTX"),
            "artifacts": {
                "ttgir_control": file_ref(SHARED / "shared-cross-lane-control.ttgir", "TTGIR_CONTROL_ANCESTOR"),
                "ptx": file_ref(SHARED / ptx_name, "FINAL_PTX"),
                "cubin": file_ref(SHARED / cubin_name, "EXACT_BINARY"),
                "snapshot": file_ref(SHARED / snapshot_name, "SNAPSHOT"),
                "pgir": file_ref(SHARED / pgir_name, "PGIR"),
                "checker": file_ref(checker_path, "CURRENT_CORE_CHECKER"),
                "current_core_replay": file_ref(SHARED_REPLAY, "CURRENT_CORE_REPLAY_RECEIPT"),
            },
            "audit": {
                "status": "CONFIRMED",
                "label": label,
                "method": "EXACT_BINARY_MANUAL_PROTOCOL_TRACE",
                "artifact": file_ref(SHARED / "shared-race-binary-pair.json", "PAIR_AUDIT"),
                "rationale": (
                    "The final-PTX positive removes exactly one compiler-generated CTA barrier "
                    "and retains an unordered cross-lane shared RAW; the control retains the barrier."
                    if role == "POSITIVE"
                    else "The exact control binary retains the CTA barrier between the shared producer and consumer."
                ),
            },
            "pipeguard": {
                "verdict": pipeguard["verdict"],
                "diagnostic": checked["primary_diagnostic"],
                "agreement": pipeguard["verdict"] == checked["expected_verdict"],
                "representation": pipeguard["statistics"]["representation"],
                "logical_events": pipeguard["statistics"]["logical_events"],
                "race_candidate_pairs": checked["race_candidate_pairs"],
                "exact_conflicting_access_pairs": checked["exact_conflicting_access_pairs"],
                "pairs_examined_before_termination": checked["pairs_examined_before_termination"],
                "materialized_event_objects": pipeguard["statistics"]["materialized_event_objects"],
            },
            "baselines": {
                "nvidia": {
                    "racecheck": {
                        **race_by_case[case_name],
                        "status": race_by_case[case_name]["classification"],
                        "denominator_eligible": True,
                    },
                    **{
                        mode: {
                            **record,
                            "status": record["classification"],
                            "denominator_eligible": mode != "synccheck" or case_name == "barrier-control",
                        }
                        for mode, record in other_by_case[case_name].items()
                    },
                },
                "triton_compiler_verifier": {
                    "status": "NOT_APPLICABLE_FINAL_PTX_MUTATION" if role == "POSITIVE" else "ACCEPTED_CONTROL_ANCESTOR",
                    "applicability": "STRUCTURAL_BASELINE_ONLY",
                },
            },
            "pipeline": {
                "compile": "ACCEPTED",
                "capture": "CAPTURED",
                "pgir": "PARSED",
                "checker": "CHECKED",
                "numeric": "PASS",
            },
            "final_figure_eligible": True,
            "supplemental_lock_sha256": hashlib.sha256(canonical_text(lock_document).encode()).hexdigest(),
        })
    return rows


def build_documents(generator: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    lock_document = build_supplemental_lock(generator)
    preliminary = load_json(PRELIMINARY)
    records = _catalog_records(preliminary) + _shared_records(lock_document)
    hazards = sum(record["evaluation_role"] == "POSITIVE" for record in records)
    controls = sum(record["evaluation_role"] == "SAFE_CONTROL" for record in records)
    agreements = sum(record["pipeguard"]["agreement"] is True for record in records)
    if (len(records), hazards, controls, agreements) != (6, 3, 3, 6):
        raise FinalCandidateError("final denominator reconciliation failed")
    matrix = {
        "schema_version": "pipeguard-suite4-final-candidate-matrix-v1",
        "status": "FINAL_CANDIDATE_COMPLETE_WITH_EXPLICIT_BASELINE_LIMITATIONS",
        "final_figure_eligible": True,
        "producer": producer_identity(generator),
        "catalog": {
            "path": relative(CATALOG),
            "sha256": EXPECTED[CATALOG],
            "case_count": 376,
            "mutant_variant_count": 84,
        },
        "lock": {"path": relative(LOCK), "sha256": EXPECTED[LOCK]},
        "baseline_index": {
            "path": relative(BASELINE_INDEX),
            "sha256": EXPECTED[BASELINE_INDEX],
            "role": "HISTORICAL_PARTIAL_COVERAGE_NOT_FINAL_ROW_INDEX",
        },
        "supplemental_input_lock": {
            "path": "corpus/mutations/suite4/supplemental-input-lock-v1.json",
            "sha256": hashlib.sha256(canonical_text(lock_document).encode()).hexdigest(),
        },
        "source_preliminary_matrix": {
            "path": relative(PRELIMINARY),
            "sha256": EXPECTED[PRELIMINARY],
        },
        "denominator": {
            "rows": 6,
            "pairs": 3,
            "positives": 3,
            "controls": 3,
            "catalog_rows": 4,
            "supplemental_rows": 2,
            "pipeguard_agreements": agreements,
            "by_dsl": {
                "tilelang": {"rows": 2, "agreements": 2},
                "triton": {"rows": 4, "agreements": 4},
            },
            "racecheck_in_domain": {
                "hazards": 1,
                "detected_hazards": 0,
                "safe_controls": 1,
                "correct_safe_controls": 1,
            },
        },
        "figure_contract": {
            "panels": ["tilelang", "triton"],
            "oracle_series": False,
            "unit": "independently-confirmed-case",
            "print_denominators": True,
        },
        "records": records,
        "rejected_labels": preliminary["rejected_labels"],
        "optional_cp_async_pair": {
            "denominator": False,
            "status": "PENDING_UNSUPPORTED",
            "static_gate": file_ref(CP_ASYNC_STATIC, "OPTIONAL_CP_ASYNC_STATIC_GATE"),
            "dynamic_model": file_ref(CP_ASYNC_DYNAMIC, "OPTIONAL_CP_ASYNC_DYNAMIC_MODEL"),
        },
        "integrity_notes": [
            "The denominator is fixed before any optional cp.async executable result.",
            "The shared pair is a supplemental final-PTX mutation/control, not a catalog DSL-source mutation.",
            "TileLang M6 RaceCheck used a non-identical baseline-era cubin and has undocumented TMA coverage; its silence is not counted as a miss.",
            "Triton M14/M15 RaceCheck and SyncCheck silence is out of domain or not applicable; initcheck contaminated timeouts are retained but excluded.",
            "Numeric PASS is recorded independently and never establishes race freedom.",
        ],
    }
    return lock_document, matrix


def write_new(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(canonical_text(value))
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def check_exact(path: Path, value: object) -> None:
    if path.read_text() != canonical_text(value):
        raise FinalCandidateError(f"generated artifact drift: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Freeze the Suite 4 final-candidate denominator")
    parser.add_argument(
        "--lock-output",
        type=Path,
        default=ROOT / "corpus/mutations/suite4/supplemental-input-lock-v1.json",
    )
    parser.add_argument(
        "--matrix-output",
        type=Path,
        default=ROOT / "corpus/mutations/suite4/final-candidate-matrix-v1.json",
    )
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    generator = Path(__file__).resolve()
    lock_document, matrix = build_documents(generator)
    if args.check:
        check_exact(args.lock_output, lock_document)
        check_exact(args.matrix_output, matrix)
    else:
        write_new(args.lock_output, lock_document)
        write_new(args.matrix_output, matrix)
    print(json.dumps({
        "lock": {"path": str(args.lock_output), "sha256": sha256_file(args.lock_output)},
        "matrix": {"path": str(args.matrix_output), "sha256": sha256_file(args.matrix_output)},
        "status": matrix["status"],
        "denominator": matrix["denominator"],
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
