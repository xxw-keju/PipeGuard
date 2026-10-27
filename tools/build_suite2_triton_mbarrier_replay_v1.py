from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
TESTS = ROOT / "tests"
sys.path[:0] = [str(ROOT / "src"), str(TESTS), str(ROOT / "tools")]

from build_suite2_records import canonical_bytes, write_new  # noqa: E402
from build_suite2_mismatch_audit import report_summary  # noqa: E402
from pipeguard.checker import check_module  # noqa: E402
from pipeguard.frontends.triton import triton_snapshot_to_module  # noqa: E402
from pipeguard.model import module_from_dict, normalize_module  # noqa: E402
from suite.pgir_direct.cases.catalog import CASES  # noqa: E402

SCHEMA_VERSION = "pipeguard-suite2-triton-mbarrier-replay-v1"
STATUS = "PRELIMINARY_REPLAY_35_CHECKED_2_DIAGNOSTIC_MISMATCHES_NOT_FINAL_FIGURE"
CAPTURE_PYTHON = Path("/usr/local/miniconda3/envs/tl-pg/bin/python")
TRITON_ROOT = ROOT.parent / "triton"
TRITON_PYTHON = TRITON_ROOT / "python"
CAPTURE_TIMEOUT_S = 60
EXPECTED_INPUTS = {
    ".artifacts/frontend-pairs/suite2-labeled-records-v1.preliminary.json":
        "6e5fddddc8fbb2a4753462b85abf74456d6fe9633e16b0dc4be1a3356182b788",
    ".artifacts/frontend-pairs/suite2-mismatch-adjudication-v1.blocked.json":
        "c28451764f0c500478f515f0d33ad4df408a162b1fd02ece0951b3303caeefcf",
    "src/pipeguard/checker.py":
        "1d0b93e6502b7a12ff00b4a1eb926617e9f8331758e4fd58057149450f3b0629",
    "src/pipeguard/direct_parametric.py":
        "22564a5d38845cd21f6ae55c91ee0d3b0975a6b9b75998285a16809c33593d1c",
    "src/pipeguard/frontends/triton.py":
        "fe8fe220b7164a9a9d46b9e6acbf042029c4ce9e3c303bd02d7f40bba7db8f04",
    "tests/suite/pgir_direct/cases/catalog.py":
        "8d53f55d768aef42cc775046a1bcb4f85d5727107caa7e52178939ff98e3e04f",
    "tools/build_suite2_mismatch_audit.py":
        "f5be407b995fb003b1183a726119032813fd7d3b69a58259d24150a6093435ec",
    "tools/build_suite2_records.py":
        "ae10489b2b450faa48ef894ef6db36494d9f97c43daec9115261a120b49fcd6b",
    "tools/triton_capture.py":
        "dc73c535531b07802e2ee680b2544929916f72f9de50633c992af88f04679ced",
}
EXPECTED_SELECTED = {
    "records": 35,
    "families": {"MB-REUSE": 24, "MBARRIER": 11},
    "oracle_verdicts": {"SAFE": 19, "SYNC_HAZARD": 16},
    "content_unique_raw": 13,
}
EXPECTED_NEW_MISMATCHES = {
    "PGIR-MB-06-NAMED-TX-BYTES-MISMATCH-BUG": {
        "oracle": ("SYNC_HAZARD", "TX_BYTES_MISMATCH"),
        "current": ("SYNC_HAZARD", "ARRIVAL_UNDERFLOW"),
    },
    "PGIR-MB-09-NAMED-STALE-GENERATION-WAIT-BUG": {
        "oracle": ("SYNC_HAZARD", "PROTOCOL_INVALID_PHASE_ADVANCE"),
        "current": ("SYNC_HAZARD", "WRONG_PARITY"),
    },
}
_CAPTURE_PROGRAM = """\
import json
import sys
from pathlib import Path
import triton_capture

snapshot = triton_capture.capture_file(
    Path(sys.argv[1]),
    source=sys.argv[2],
    command=sys.argv[3],
    grid_ctas=1,
)
print(json.dumps(snapshot, sort_keys=True))
"""


class Suite2MbarrierReplayError(RuntimeError):
    pass


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(ROOT.resolve()))


def file_ref(path: Path) -> dict[str, Any]:
    data = path.read_bytes()
    return {"path": _relative(path), "bytes": len(data), "sha256": sha256_bytes(data)}


def external_file_ref(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    data = resolved.read_bytes()
    return {"path": str(resolved), "bytes": len(data), "sha256": sha256_bytes(data)}


def verify_inputs() -> dict[str, dict[str, Any]]:
    refs = {}
    for relative, expected in EXPECTED_INPUTS.items():
        path = ROOT / relative
        if not path.is_file():
            raise Suite2MbarrierReplayError(f"missing frozen input: {relative}")
        ref = file_ref(path)
        if ref["sha256"] != expected:
            raise Suite2MbarrierReplayError(
                f"frozen input drift: {relative}: expected {expected}, got {ref['sha256']}"
            )
        refs[relative] = ref
    return refs


def _diagnostic_code(record: dict[str, Any]) -> str | None:
    diagnostic = record["pipeguard"].get("diagnostic")
    if isinstance(diagnostic, dict):
        return diagnostic.get("code")
    return diagnostic


def select_records(preliminary: dict[str, Any]) -> list[dict[str, Any]]:
    selected = [
        record
        for record in preliminary["records"]
        if record["dsl"] == "triton"
        and record["oracle"]["stage"] == "CHECKER"
        and record["family"] in {"MBARRIER", "MB-REUSE"}
        and record["pipeguard"]["status"] == "UNSUPPORTED"
        and _diagnostic_code(record) == "TRITON_WAIT_PHASE_UNRESOLVED"
    ]
    selected.sort(key=lambda item: item["label"])
    observed = {
        "records": len(selected),
        "families": dict(sorted(Counter(item["family"] for item in selected).items())),
        "oracle_verdicts": dict(sorted(Counter(
            item["oracle"]["expected_verdict"] for item in selected
        ).items())),
        "content_unique_raw": len({
            item["kernel_identity"]["sha256"] for item in selected
        }),
    }
    if observed != EXPECTED_SELECTED:
        raise Suite2MbarrierReplayError(
            f"selected-row inventory drift: expected {EXPECTED_SELECTED}, got {observed}"
        )
    return selected


def capture_snapshot(raw_path: Path, label: str) -> dict[str, Any]:
    if not CAPTURE_PYTHON.is_file() or not TRITON_PYTHON.is_dir():
        raise Suite2MbarrierReplayError("pinned Triton capture environment is unavailable")
    environment = os.environ.copy()
    environment.update({
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONPATH": os.pathsep.join((str(TRITON_PYTHON), str(ROOT / "tools"))),
    })
    try:
        process = subprocess.run(
            [
                str(CAPTURE_PYTHON),
                "-c",
                _CAPTURE_PROGRAM,
                str(raw_path),
                f"suite2-replay:{label}",
                "suite2-frozen-raw-current-parser-replay",
            ],
            cwd=ROOT,
            env=environment,
            check=False,
            capture_output=True,
            text=True,
            timeout=CAPTURE_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired as error:
        raise Suite2MbarrierReplayError(f"capture timeout for {label}") from error
    if process.returncode != 0:
        raise Suite2MbarrierReplayError(
            f"capture failed for {label}: exit={process.returncode}: {process.stderr[-2000:]}"
        )
    try:
        snapshot = json.loads(process.stdout)
    except json.JSONDecodeError as error:
        raise Suite2MbarrierReplayError(
            f"capture emitted invalid JSON for {label}: {process.stdout[-1000:]}"
        ) from error
    if not isinstance(snapshot, dict):
        raise Suite2MbarrierReplayError(f"capture did not emit an object for {label}")
    return snapshot


def _wait_evidence(snapshot: dict[str, Any], label: str) -> list[dict[str, Any]]:
    waits = []
    for operation in snapshot.get("operations", []):
        if operation.get("op") != "triton.mbarrier.wait_parity":
            continue
        parity = operation.get("raw_parity")
        if isinstance(parity, bool) or not isinstance(parity, int):
            raise Suite2MbarrierReplayError(f"unresolved wait parity after capture for {label}")
        evidence = operation.get("execution_evidence")
        if not isinstance(evidence, dict):
            raise Suite2MbarrierReplayError(f"missing wait execution evidence for {label}")
        waits.append({
            key: deepcopy(operation[key])
            for key in (
                "id", "resource", "lifecycle_epoch", "generation", "raw_parity",
                "logical_stream_id", "stream_rank", "polling_coalesced",
                "execution_evidence",
            )
            if key in operation
        })
    if not waits:
        raise Suite2MbarrierReplayError(f"no captured wait evidence for {label}")
    return waits


def _git_identity(root: Path) -> dict[str, Any]:
    def command(*args: str) -> bytes:
        return subprocess.check_output(args, cwd=root)
    return {
        "root": str(root.resolve()),
        "head": command("git", "rev-parse", "HEAD").decode().strip(),
        "tracked_binary_diff_sha256": sha256_bytes(
            command("git", "diff", "--binary", "HEAD", "--")
        ),
        "status_digest": sha256_bytes(
            command("git", "status", "--porcelain=v1", "-z")
        ),
        "untracked_list_digest": sha256_bytes(
            command("git", "ls-files", "--others", "--exclude-standard", "-z")
        ),
    }


def _capture_runtime_identity() -> dict[str, Any]:
    return {
        "python": external_file_ref(CAPTURE_PYTHON),
        "triton_repository": _git_identity(TRITON_ROOT),
        "triton_init": external_file_ref(TRITON_PYTHON / "triton/__init__.py"),
        "libtriton": external_file_ref(TRITON_PYTHON / "triton/_C/libtriton.so"),
        "version": "3.8.0",
        "dependency_boundary": (
            "capture runs under the pinned Python 3.10 Triton source tree; frontend conversion "
            "and checking run in the caller's PipeGuard Python environment"
        ),
    }


def build_replay() -> dict[str, Any]:
    bindings = verify_inputs()
    preliminary = json.loads(
        (ROOT / ".artifacts/frontend-pairs/suite2-labeled-records-v1.preliminary.json").read_text()
    )
    mismatch_audit = json.loads(
        (ROOT / ".artifacts/frontend-pairs/suite2-mismatch-adjudication-v1.blocked.json").read_text()
    )
    if mismatch_audit["summary"]["retained_oracle_mismatches"] != 6:
        raise Suite2MbarrierReplayError("frozen mismatch denominator is not six")
    selected = select_records(preliminary)
    cases = {case.label: case for case in CASES}
    rows = []
    for record in selected:
        label = record["label"]
        raw_path = ROOT / record["capture"]["raw"]["path"]
        raw_ref = file_ref(raw_path)
        if raw_ref["sha256"] != record["kernel_identity"]["sha256"]:
            raise Suite2MbarrierReplayError(f"raw source drift for {label}")
        snapshot = capture_snapshot(raw_path, label)
        snapshot_bytes = canonical_bytes(snapshot)
        waits = _wait_evidence(snapshot, label)
        module = triton_snapshot_to_module(snapshot)
        normalized_pgir = normalize_module(module)
        current_report = check_module(module, engine="maximal")
        current = report_summary(current_report)

        case = cases.get(label)
        if case is None:
            raise Suite2MbarrierReplayError(f"direct reference missing for {label}")
        direct_report = check_module(
            module_from_dict(deepcopy(case.raw)), engine="crosscheck"
        )
        direct = report_summary(direct_report)
        oracle = {
            "expected_verdict": record["oracle"]["expected_verdict"],
            "expected_diagnostic": record["oracle"]["expected_diagnostic"],
        }
        direct_pair = (direct["verdict"], direct["primary_diagnostic"])
        oracle_pair = (oracle["expected_verdict"], oracle["expected_diagnostic"])
        if direct_pair != oracle_pair:
            raise Suite2MbarrierReplayError(
                f"direct reference no longer matches Oracle for {label}: {direct_pair} != {oracle_pair}"
            )
        current_pair = (current["verdict"], current["primary_diagnostic"])
        rows.append({
            "record_id": record["record_id"],
            "label": label,
            "family": record["family"],
            "pair_id": record["pair_id"],
            "polarity": record["polarity"],
            "variant": record["variant"],
            "raw_ttgir": raw_ref,
            "capture": {
                "status": "CAPTURED_CURRENT_PINNED_TRITON",
                "snapshot_sha256": sha256_bytes(snapshot_bytes),
                "snapshot_bytes": len(snapshot_bytes),
                "operations": len(snapshot.get("operations", [])),
                "waits": waits,
            },
            "pgir": {
                "status": "CONVERTED_CURRENT_FRONTEND",
                "sha256": sha256_bytes(canonical_bytes(normalized_pgir)),
                "schema_version": normalized_pgir.get("schema_version"),
            },
            "oracle": oracle,
            "direct_reference_replay": direct,
            "current_frontend_replay": current,
            "strict_oracle_agreement": current_pair == oracle_pair,
            "disposition": (
                "CHECKED_STRICT_ORACLE_MATCH"
                if current_pair == oracle_pair
                else "CHECKED_DIAGNOSTIC_MISMATCH_RETAINED"
            ),
            "final_figure_eligible": False,
        })

    matches = [row for row in rows if row["strict_oracle_agreement"]]
    mismatches = [row for row in rows if not row["strict_oracle_agreement"]]
    observed_mismatches = {
        row["label"]: {
            "oracle": (
                row["oracle"]["expected_verdict"],
                row["oracle"]["expected_diagnostic"],
            ),
            "current": (
                row["current_frontend_replay"]["verdict"],
                row["current_frontend_replay"]["primary_diagnostic"],
            ),
        }
        for row in mismatches
    }
    if observed_mismatches != EXPECTED_NEW_MISMATCHES:
        raise Suite2MbarrierReplayError(
            f"replay mismatch inventory drift: {observed_mismatches}"
        )
    original = preliminary["summary"]
    overlay = {
        "checker_stage_records": original["checker_stage_records"],
        "pipeguard_checked_checker_records":
            original["pipeguard_checked_checker_records"] + len(rows),
        "pipeguard_matching_checker_records":
            original["pipeguard_matching_checker_records"] + len(matches),
        "pipeguard_mismatch_checker_records":
            original["pipeguard_mismatch_checker_records"] + len(mismatches),
        "pipeguard_incomplete_checker_records":
            original["pipeguard_incomplete_checker_records"] - len(rows),
    }
    if overlay != {
        "checker_stage_records": 560,
        "pipeguard_checked_checker_records": 395,
        "pipeguard_matching_checker_records": 387,
        "pipeguard_mismatch_checker_records": 8,
        "pipeguard_incomplete_checker_records": 165,
    }:
        raise Suite2MbarrierReplayError(f"overlay accounting drift: {overlay}")

    return {
        "schema_version": SCHEMA_VERSION,
        "status": STATUS,
        "evidence_level": "PRELIMINARY_CURRENT_TOOL_REPLAY",
        "final_figure_eligible": False,
        "all_correct": False,
        "scope": {
            "dsl": "triton",
            "oracle_stage": "CHECKER",
            "families": ["MB-REUSE", "MBARRIER"],
            "historical_status": "UNSUPPORTED",
            "historical_diagnostic": "TRITON_WAIT_PHASE_UNRESOLVED",
            "selection_policy": (
                "complete frozen 35-row cohort; no failing or mismatching row is omitted"
            ),
        },
        "input_bindings": bindings,
        "builder": file_ref(Path(__file__)),
        "current_consumer_identity": _git_identity(ROOT),
        "capture_runtime_identity": _capture_runtime_identity(),
        "producer_provenance_limitation": (
            "This replay binds the current capture/runtime/checker identities and frozen raw TTGIR. "
            "It does not recover or relabel the unavailable original Suite 2 producer identity."
        ),
        "summary": {
            "selected_records": len(rows),
            "content_unique_raw_ttgir": len({row["raw_ttgir"]["sha256"] for row in rows}),
            "captured": len(rows),
            "converted": len(rows),
            "checked": len(rows),
            "strict_oracle_matches": len(matches),
            "strict_oracle_mismatches": len(mismatches),
            "direct_reference_oracle_matches": len(rows),
            "families": dict(sorted(Counter(row["family"] for row in rows).items())),
            "oracle_verdicts": dict(sorted(Counter(
                row["oracle"]["expected_verdict"] for row in rows
            ).items())),
        },
        "preliminary_overlay_accounting": overlay,
        "new_mismatches": [
            {
                "label": row["label"],
                "oracle": row["oracle"],
                "current_frontend_replay": row["current_frontend_replay"],
                "disposition": row["disposition"],
            }
            for row in mismatches
        ],
        "remaining_blockers": [
            "BASELINES_NOT_RUN_NO_EXECUTABLE_BINARY",
            "PIPEGUARD_INCOMPLETE_CHECKER_ROWS_165",
            "PIPEGUARD_ORACLE_MISMATCHES_8",
            "CELL_MINIMUM_15_CONTENT_UNIQUE_KERNELS_NOT_MET",
            "ORIGINAL_PRODUCER_IDENTITY_ABSENT_FROM_SOURCE_ARTIFACTS",
            "FRONTEND_SCENARIOS_NOT_EQUIVALENT_6",
            "REPLAY_DIAGNOSTIC_MISMATCHES_2",
        ],
        "rows": rows,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    try:
        payload = build_replay()
        expected = canonical_bytes(payload)
        if args.check:
            if not args.output.is_file():
                raise Suite2MbarrierReplayError(f"output missing: {args.output}")
            actual = args.output.read_bytes()
            if actual != expected:
                raise Suite2MbarrierReplayError(
                    f"output mismatch: expected {sha256_bytes(expected)}, got {sha256_bytes(actual)}"
                )
        else:
            write_new(args.output, payload)
        print(json.dumps({
            "status": payload["status"],
            **payload["summary"],
            **payload["preliminary_overlay_accounting"],
            "final_figure_eligible": payload["final_figure_eligible"],
        }, sort_keys=True))
        return 0
    except (OSError, ValueError, subprocess.SubprocessError, Suite2MbarrierReplayError) as error:
        print(f"suite2 Triton mbarrier replay error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
