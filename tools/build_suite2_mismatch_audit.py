from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path[:0] = [str(ROOT / "src"), str(TESTS), str(ROOT / "tools")]

from build_suite2_records import canonical_bytes, write_new  # noqa: E402
from pipeguard.checker import check_module  # noqa: E402
from pipeguard.model import module_from_dict  # noqa: E402
from suite.pgir_direct.cases.catalog import CASES  # noqa: E402

SCHEMA_VERSION = "pipeguard-suite2-mismatch-adjudication-v1"
STATUS = "BLOCKED_NOT_FINAL_FIGURE"
EXPECTED_BINDINGS = {
    ".artifacts/frontend-pairs/suite2-labeled-records-v1.preliminary.json": "6e5fddddc8fbb2a4753462b85abf74456d6fe9633e16b0dc4be1a3356182b788",
    ".artifacts/frontend-pairs/tilelang-checks.json": "99c5da7c2f5442acfa4fb3d923ce333efc729bf91f205c5a8fca2f872421ac3f",
    ".artifacts/frontend-pairs/triton-checks.json": "af3cc960803f39fe346b4693e70a5efb135a975ba319dacf0f22bdd17780ddd2",
    "src/pipeguard/checker.py": "1d0b93e6502b7a12ff00b4a1eb926617e9f8331758e4fd58057149450f3b0629",
    "src/pipeguard/direct_parametric.py": "22564a5d38845cd21f6ae55c91ee0d3b0975a6b9b75998285a16809c33593d1c",
    "src/pipeguard/engine.py": "47f6bf456a31a2e17b75525248428d6d5785460776e71afa0ad9b3fb1e0e096e",
    "tests/suite/frontend_pairs/catalog.py": "e1e3514f555f78e61f051ed0f62c5f1a6656cdb5409dcdb6c61a3571625c0e3c",
    "tests/suite/frontend_pairs/sources/tilelang_cases.py": "e90f8b8d6a0dab7e0ab6b19060d78a124da5784f339f8addc09b52857ce7b4b9",
    "tests/suite/frontend_pairs/sources/triton_cases.py": "586c175ad6d519476105c3ee7bfea22ce97a4041bf02893cb478744673bc0898",
    "tests/suite/pgir_direct/cases/catalog.py": "8d53f55d768aef42cc775046a1bcb4f85d5727107caa7e52178939ff98e3e04f",
    "tests/test_build_suite2_records.py": "af4b385a803455315bfec4a13db7a6aa05dfe5ee0e30eb7752ea4184b32707d5",
    "tools/build_suite2_records.py": "ae10489b2b450faa48ef894ef6db36494d9f97c43daec9115261a120b49fcd6b",
    ".artifacts/frontend-pairs/pgir/tilelang/PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG.json": "5b382cb2245d2952de4f6b0ecc4dc699b2235c125659a88d2050d95a863f4678",
    ".artifacts/frontend-pairs/pgir/tilelang/PGIR-MB-01-NAMED-REINITIALIZE-LIFECYCLE-BUG.json": "b3d3769a242ad795c46a4201e80d94879758b4e6a64f62191b88020336d934b7",
    ".artifacts/frontend-pairs/pgir/tilelang/PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG.json": "c05bc425ccb0d584a3c3d21caa846951ba14d0dfb50f967ae7026800e57a08d1",
    ".artifacts/frontend-pairs/pgir/triton/PGIR-BAR-01-NAMED-MISSING-PARTICIPANT-BUG.json": "217507556e82d796f04a27ea2c81096b6b2cf104c52d70fa5bcf9034886a62cc",
    ".artifacts/frontend-pairs/pgir/triton/PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG.json": "c2072f05953ba0a5f9820b107af03c7e3b64c37853bd4b16c08b24afe45e3167",
    ".artifacts/frontend-pairs/pgir/triton/PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG.json": "0dc7e49ce4d68521cc8fd188a82e250a93d174f0293f7d4c3b9412a0d8bce3e7",
}
ROWS = (
    {
        "dsl": "tilelang",
        "label": "PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG",
        "current": ("SYNC_HAZARD", "MISSING_BARRIER_PARTICIPANT"),
        "source_lines": "tests/suite/frontend_pairs/sources/tilelang_cases.py:276-279",
        "reason": "frontend emits one complete CTA synchronization followed by a distinct lane-0-only synchronization; it does not duplicate one owner's arrival in the same resource invocation",
    },
    {
        "dsl": "tilelang",
        "label": "PGIR-MB-01-NAMED-REINITIALIZE-LIFECYCLE-BUG",
        "current": ("SAFE", None),
        "source_lines": "tests/suite/frontend_pairs/sources/tilelang_cases.py:291-317",
        "reason": "frontend emits one initialization and one generation-0 arrival; it emits no second initialization, invalidate, or later-generation reuse",
    },
    {
        "dsl": "tilelang",
        "label": "PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG",
        "current": ("DATA_HAZARD", "UNORDERED_CONFLICT"),
        "source_lines": "tests/suite/frontend_pairs/sources/tilelang_cases.py:424-428",
        "reason": "frontend wait occurs before issue; the direct reference instead stages an issue and then waits without a commit",
    },
    {
        "dsl": "triton",
        "label": "PGIR-BAR-01-NAMED-MISSING-PARTICIPANT-BUG",
        "current": ("SAFE", None),
        "source_lines": "tests/suite/frontend_pairs/sources/triton_cases.py:225-246",
        "reason": "frontend emits one complete aggregate CTA barrier and has no missing-participant branch",
    },
    {
        "dsl": "triton",
        "label": "PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG",
        "current": ("SAFE", None),
        "source_lines": "tests/suite/frontend_pairs/sources/triton_cases.py:231-246",
        "reason": "frontend emits two complete barrier invocations, not duplicate arrivals to one invocation",
    },
    {
        "dsl": "triton",
        "label": "PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG",
        "current": ("DATA_HAZARD", "UNORDERED_CONFLICT"),
        "source_lines": "tests/suite/frontend_pairs/sources/triton_cases.py:447-450",
        "reason": "frontend wait occurs before issue and commit; the direct reference instead stages an issue and then waits without a commit",
    },
)


class MismatchAuditError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {
        "path": str(path.resolve().relative_to(ROOT.resolve())),
        "bytes": len(data),
        "sha256": sha256_bytes(data),
    }


def verify_bindings() -> dict[str, dict[str, Any]]:
    result = {}
    for relative, expected in EXPECTED_BINDINGS.items():
        path = ROOT / relative
        if not path.is_file():
            raise MismatchAuditError(f"missing frozen input: {relative}")
        ref = file_ref(path)
        if ref["sha256"] != expected:
            raise MismatchAuditError(
                f"frozen input drift: {relative}: expected {expected}, got {ref['sha256']}"
            )
        result[relative] = ref
    return result


def primary_diagnostic(report: dict[str, Any]) -> str | None:
    diagnostics = report.get("diagnostics", [])
    return diagnostics[0].get("code") if diagnostics else None


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    summary = {
        "verdict": report["verdict"],
        "primary_diagnostic": primary_diagnostic(report),
    }
    if report.get("diagnostics"):
        diagnostic = report["diagnostics"][0]
        summary["diagnostic_evidence"] = {
            key: deepcopy(diagnostic[key])
            for key in ("code", "event", "events", "resource")
            if key in diagnostic
        }
    if report.get("witness"):
        witness = report["witness"]
        summary["witness_evidence"] = {
            key: deepcopy(witness[key])
            for key in ("category", "event", "events", "allocation_id", "overlap", "effects")
            if key in witness
        }
    return summary


def event_inventory(raw: dict[str, Any]) -> dict[str, Any]:
    events = raw.get("events", [])
    selected_keys = (
        "id", "type", "actor", "seq", "resource", "invocation_id",
        "lifecycle_epoch", "generation", "endpoint",
    )
    return {
        "event_count": len(events),
        "types": dict(sorted(Counter(event.get("type", "<missing>") for event in events).items())),
        "protocol_events": [
            {key: event[key] for key in selected_keys if key in event}
            for event in events
            if any(token in event.get("type", "") for token in ("Barrier", "Tma", "TMA", "MemWrite"))
        ],
    }


def load_checks(dsl: str) -> dict[str, dict[str, Any]]:
    records = json.loads(
        (ROOT / f".artifacts/frontend-pairs/{dsl}-checks.json").read_text()
    )
    return {record["label"]: record for record in records}


def build_audit(artifacts: Path) -> dict[str, Any]:
    artifacts = artifacts.resolve()
    if artifacts != (ROOT / ".artifacts/frontend-pairs").resolve():
        raise MismatchAuditError(
            "this frozen v1 audit accepts only .artifacts/frontend-pairs"
        )
    bindings = verify_bindings()
    preliminary = json.loads(
        (artifacts / "suite2-labeled-records-v1.preliminary.json").read_text()
    )
    expected_accounting = {
        "records": 580,
        "checker_stage_records": 560,
        "pipeguard_checked_checker_records": 360,
        "pipeguard_matching_checker_records": 354,
        "pipeguard_mismatch_checker_records": 6,
        "pipeguard_incomplete_checker_records": 200,
    }
    for key, expected in expected_accounting.items():
        if preliminary["summary"].get(key) != expected:
            raise MismatchAuditError(f"preliminary accounting drift: {key}")
    mismatch_keys = {
        (record["dsl"], record["label"])
        for record in preliminary["mismatches"]
    }
    row_keys = {(row["dsl"], row["label"]) for row in ROWS}
    if mismatch_keys != row_keys or len(row_keys) != 6:
        raise MismatchAuditError("mismatch inventory is not the frozen six-row set")

    cases = {case.label: case for case in CASES}
    distinct_labels = sorted({row["label"] for row in ROWS})
    direct_replays = {}
    for label in distinct_labels:
        case = cases[label]
        report = check_module(module_from_dict(deepcopy(case.raw)), engine="crosscheck")
        observed = (report["verdict"], primary_diagnostic(report))
        expected = (case.expected_verdict, case.expected_diagnostic)
        if observed != expected:
            raise MismatchAuditError(
                f"direct reference no longer matches Oracle for {label}: {observed} != {expected}"
            )
        direct_replays[label] = {
            "oracle": {
                "expected_verdict": case.expected_verdict,
                "expected_diagnostic": case.expected_diagnostic,
                "catalog_lines": {
                    "PGIR-BAR-03-NAMED-DUPLICATE-ARRIVAL-BUG": "tests/suite/pgir_direct/cases/catalog.py:959-994",
                    "PGIR-MB-01-NAMED-REINITIALIZE-LIFECYCLE-BUG": "tests/suite/pgir_direct/cases/catalog.py:1578-1590",
                    "PGIR-TMA-05-NAMED-STORE-WAIT-BEFORE-COMMIT-BUG": "tests/suite/pgir_direct/cases/catalog.py:1401-1425",
                    "PGIR-BAR-01-NAMED-MISSING-PARTICIPANT-BUG": "tests/suite/pgir_direct/cases/catalog.py:1532-1550",
                }[label],
            },
            "replay": report_summary(report),
            "raw_canonical_sha256": sha256_bytes(canonical_bytes(case.raw)),
            "event_inventory": event_inventory(case.raw),
        }

    checks = {dsl: load_checks(dsl) for dsl in ("tilelang", "triton")}
    adjudications = []
    for row in ROWS:
        dsl, label = row["dsl"], row["label"]
        pgir_relative = f"pgir/{dsl}/{label}.json"
        pgir_raw = json.loads((artifacts / pgir_relative).read_text())
        current_report = check_module(module_from_dict(deepcopy(pgir_raw)), engine="maximal")
        current = (current_report["verdict"], primary_diagnostic(current_report))
        if current != row["current"]:
            raise MismatchAuditError(
                f"frontend replay drift for {dsl}/{label}: {current} != {row['current']}"
            )
        frozen = checks[dsl][label]
        frozen_result = (
            frozen.get("actual_verdict"), frozen.get("actual_diagnostic")
        )
        stored_current_match = frozen_result == current
        stale_expected = (
            dsl == "tilelang"
            and label == "PGIR-MB-01-NAMED-REINITIALIZE-LIFECYCLE-BUG"
        )
        if stored_current_match == stale_expected:
            raise MismatchAuditError(
                f"unexpected frozen/current relation for {dsl}/{label}"
            )
        direct = direct_replays[label]["replay"]
        if current == (direct["verdict"], direct["primary_diagnostic"]):
            raise MismatchAuditError(
                f"frontend unexpectedly became direct-reference equivalent: {dsl}/{label}"
            )
        adjudications.append({
            "record_id": f"suite2-mismatch:{dsl}:{label}",
            "dsl": dsl,
            "label": label,
            "source_evidence": {
                "frontend_lines": row["source_lines"],
                "frontend_source": bindings[
                    f"tests/suite/frontend_pairs/sources/{dsl}_cases.py"
                ],
                "frontend_pgir": bindings[f".artifacts/frontend-pairs/{pgir_relative}"],
            },
            "direct_reference": direct_replays[label],
            "frozen_check": {
                "source": bindings[f".artifacts/frontend-pairs/{dsl}-checks.json"],
                "status": frozen["status"],
                "actual_verdict": frozen.get("actual_verdict"),
                "actual_diagnostic": frozen.get("actual_diagnostic"),
                "matches_expected": frozen.get("matches_expected"),
            },
            "current_frontend_replay": report_summary(current_report),
            "frontend_event_inventory": event_inventory(pgir_raw),
            "frozen_current_relation": (
                "FROZEN_CHECK_RECORD_MATCHES_CURRENT_CORE"
                if stored_current_match
                else "FROZEN_CHECK_RECORD_STALE_AGAINST_CURRENT_CORE"
            ),
            "semantic_equivalence": {
                "equivalent_to_direct_reference": False,
                "status": "FRONTEND_SCENARIO_NOT_EQUIVALENT_TO_DIRECT_REFERENCE",
                "reason": row["reason"],
            },
            "checker_regression_demonstrated": False,
            "denominator_disposition": "RETAINED_MISMATCH_BLOCKS_ALL_CORRECT",
        })

    stale = sum(
        record["frozen_current_relation"]
        == "FROZEN_CHECK_RECORD_STALE_AGAINST_CURRENT_CORE"
        for record in adjudications
    )
    matched = len(adjudications) - stale
    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "evidence_level": "PRELIMINARY_ADJUDICATION",
        "final_figure_eligible": False,
        "all_correct": False,
        "input_bindings": bindings,
        "builder": file_ref(Path(__file__)),
        "preliminary_accounting": {
            key: preliminary["summary"][key] for key in expected_accounting
        },
        "summary": {
            "adjudications": len(adjudications),
            "distinct_direct_reference_labels": len(direct_replays),
            "direct_reference_oracle_matches": len(direct_replays),
            "frontend_direct_semantic_divergences": len(adjudications),
            "frozen_current_matches": matched,
            "frozen_current_stale": stale,
            "checker_regressions_demonstrated": 0,
            "retained_oracle_mismatches": len(adjudications),
        },
        "interpretation": (
            "The direct-reference Oracle remains valid. These six frontend rows are not "
            "semantic realizations of their direct-reference scenarios, so they remain "
            "mismatches and cannot support an all-correct or final-figure claim."
        ),
        "direct_reference_replays": direct_replays,
        "adjudications": adjudications,
        "remaining_blockers": preliminary["final_blockers"] + [
            "FRONTEND_SCENARIOS_NOT_EQUIVALENT_6"
        ],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = build_audit(args.artifacts)
        expected = canonical_bytes(payload)
        if args.check:
            if not args.output.is_file():
                raise MismatchAuditError(f"output missing: {args.output}")
            actual = args.output.read_bytes()
            if actual != expected:
                raise MismatchAuditError(
                    f"output mismatch: expected {sha256_bytes(expected)}, got {sha256_bytes(actual)}"
                )
        else:
            write_new(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            **payload["summary"],
            "final_figure_eligible": payload["final_figure_eligible"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, MismatchAuditError) as error:
        print(f"suite2 mismatch audit error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
