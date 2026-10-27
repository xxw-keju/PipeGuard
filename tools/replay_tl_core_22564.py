from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
CORE_SHA256 = "22564a5d38845cd21f6ae55c91ee0d3b0975a6b9b75998285a16809c33593d1c"
SOURCE_MATRIX = ROOT / "corpus/mutations/suite4/final-candidate-matrix-v1.json"
SOURCE_MATRIX_SHA256 = "645317867a525e36a5d316b801e18d17617c11b7b55dad049e1cb3fc882393c7"
SYNC_MANIFEST = ROOT / ".artifacts/suite5-inputs/triton-attention-static-noncausal/manifest.json"
OUTPUT_ROOT = ROOT / ".artifacts/final-checker/tl-core-22564-v1"
MATRIX_OUTPUT = ROOT / "corpus/mutations/suite4/final-candidate-matrix-22564a5d-v1.json"
GUARDS = {
    "corpus/mutations/suite4/final-candidate-matrix-v1.json": SOURCE_MATRIX_SHA256,
    "corpus/mutations/suite4/supplemental-input-lock-v1.json": "1ad149692afb24f1b34a4fb407d588236aa16adbc29dbbf42b55b7b7e065a5dc",
    ".artifacts/final-suite4-gate/final-candidate-v1/shared-current-core-replay.json": "7ef33c1eb9a94e11636b73db06b115047005b2450993812be72c63fc060756fc",
    "tools/build_suite4_final_candidate.py": "41953fe67dc09fbfa33e4a2fd671cb37bb7d0bb7944157c38444101995b5cb33",
    "corpus/mutations/suite4/catalog.json": "127d7c6759d7da49dae1af2c65b9d48086ca5c34ac37609ce06c466388a89998",
    "corpus/mutations/suite4/lock.json": "74e27e3b8e2d3de1e23b3b9b6bc51cfc9a4c85940f8b0164ec5cbcf20bee6f54",
    "corpus/mutations/suite4/baseline-index.json": "21a172daf61fc9de975e3677e7bed29d03f00287deca977c9856da3ec5bd796a",
}
SYNC_EXPECTED = {
    "seq64": ("DATA_HAZARD", None, 7, None),
    "seq128": ("SYNC_HAZARD", "INVALIDATE_INCOMPLETE", 21, 0),
    "seq192": ("SYNC_HAZARD", "INVALIDATE_INCOMPLETE", 43, 0),
    "seq256": ("SYNC_HAZARD", "INVALIDATE_INCOMPLETE", 79, 0),
    "seq320": ("SYNC_HAZARD", "INVALIDATE_INCOMPLETE", 123, 0),
}
SUITE4_EXPECTED = {
    "triton-dense-gemm-source-m14-cross-warp-roundtrip--triton-dense-gemm-workload-original": ("DATA_HAZARD", "UNORDERED_CONFLICT", 2212, 1),
    "triton-dense-gemm-source-m15-cross-warp-barrier-control--triton-dense-gemm-workload-original": ("SAFE", None, 2212, 2212),
    "tilelang-dense-gemm-source-m6-a-before--tilelang-dense-gemm-workload-original": ("DATA_HAZARD", "UNORDERED_CONFLICT", 58840, 49273),
    "tilelang-dense-gemm-source-original--tilelang-dense-gemm-workload-original": ("SAFE", None, 54168, 54168),
    "shared-no-barrier-hazard": ("DATA_HAZARD", "UNORDERED_CONFLICT", 127, 1),
    "shared-barrier-control": ("SAFE", None, 127, 127),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_bytes(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def write_new(path: Path, value: object) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(descriptor, "wb") as output:
        output.write(canonical_bytes(value))


def ref(path: Path, kind: str | None = None) -> dict[str, Any]:
    value: dict[str, Any] = {
        "path": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
        "sha256": sha256(path),
        "bytes": path.stat().st_size,
    }
    if kind:
        value["kind"] = kind
    return value


def resolve(path: str) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def git(*args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(ROOT), *args])


def validate_guards() -> None:
    core = ROOT / "src/pipeguard/direct_parametric.py"
    if sha256(core) != CORE_SHA256:
        raise RuntimeError(f"successor core mismatch: {sha256(core)}")
    for relative, expected in GUARDS.items():
        actual = sha256(ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"guard mismatch for {relative}: {actual} != {expected}")


def report_summary(report: dict[str, Any]) -> dict[str, Any]:
    diagnostics = report.get("diagnostics") or []
    statistics = report.get("statistics") or {}
    timing = report.get("timing") or report.get("timings") or {}
    return {
        "verdict": report.get("verdict"),
        "diagnostic": diagnostics[0].get("code") if diagnostics else None,
        "exact_conflicting_access_pairs": statistics.get("exact_conflicting_access_pairs"),
        "race_candidate_pairs": statistics.get("race_candidate_pairs"),
        "pairs_examined_before_termination": statistics.get("pairs_examined_before_termination"),
        "exact_r_complete": statistics.get("exact_r_complete"),
        "race_scan_scope": statistics.get("race_scan_scope"),
        "race_ordering_performed": statistics.get("race_ordering_performed"),
        "race_scan_s": statistics.get("race_scan_s", timing.get("race_scan_s")),
        "logical_events": statistics.get("logical_events"),
        "materialized_event_objects": statistics.get("materialized_event_objects"),
        "representation": statistics.get("representation"),
    }


def require_summary(case_id: str, summary: dict[str, Any], expected: tuple[str, str | None, int, int | None]) -> None:
    verdict, diagnostic, exact_r, examined = expected
    if summary["verdict"] != verdict:
        raise RuntimeError(f"{case_id}: verdict {summary['verdict']} != {verdict}")
    if diagnostic is not None and summary["diagnostic"] != diagnostic:
        raise RuntimeError(f"{case_id}: diagnostic {summary['diagnostic']} != {diagnostic}")
    if summary["exact_conflicting_access_pairs"] != exact_r or summary["race_candidate_pairs"] != exact_r:
        raise RuntimeError(f"{case_id}: exact R mismatch")
    if examined is not None and summary["pairs_examined_before_termination"] != examined:
        raise RuntimeError(f"{case_id}: examined mismatch")
    if summary["exact_r_complete"] is not True:
        raise RuntimeError(f"{case_id}: exact_r_complete is not true")
    if verdict == "SYNC_HAZARD":
        if summary["race_scan_scope"] != "DECLARED_ACCESS_CONFLICT_ENUMERATION":
            raise RuntimeError(f"{case_id}: wrong race scan scope")
        if summary["race_ordering_performed"] is not False:
            raise RuntimeError(f"{case_id}: ordering unexpectedly performed")
        if not isinstance(summary["race_scan_s"], (int, float)) or summary["race_scan_s"] <= 0:
            raise RuntimeError(f"{case_id}: race timing is not positive")


def run_checker(pgir: Path, stem: str, output_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    request_path = output_root / f"{stem}.request.json"
    report_path = output_root / f"{stem}.checker.json"
    write_new(request_path, {"pgir_path": str(pgir), "engine": "maximal", "report_path": str(report_path)})
    environment = {
        **os.environ,
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tests"), str(ROOT / "tools"))),
        "PYTHONNOUSERSITE": "1",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
    }
    subprocess.run(["/usr/bin/python3", str(ROOT / "tools/upstream_check.py"), "--worker-request", str(request_path)], cwd=ROOT, env=environment, check=True)
    report = json.loads(report_path.read_text())
    return report_summary(report), {
        "pgir": ref(pgir, "FROZEN_PGIR"),
        "request": ref(request_path, "SUCCESSOR_REQUEST"),
        "checker": ref(report_path, "SUCCESSOR_CHECKER"),
    }


def successor_record(source_record: dict[str, Any], summary: dict[str, Any], checker_ref: dict[str, Any]) -> dict[str, Any]:
    record = copy.deepcopy(source_record)
    record["artifacts"]["predecessor_checker"] = copy.deepcopy(record["artifacts"]["checker"])
    record["artifacts"]["checker"] = checker_ref
    expected_verdict = "SAFE" if record["audit"]["label"] == "CORRECT" else record["audit"]["label"]
    record["pipeguard"] = {
        "verdict": summary["verdict"], "diagnostic": summary["diagnostic"],
        "agreement": summary["verdict"] == expected_verdict,
        "exact_conflicting_access_pairs": summary["exact_conflicting_access_pairs"],
        "race_candidate_pairs": summary["race_candidate_pairs"],
        "pairs_examined_before_termination": summary["pairs_examined_before_termination"],
        "logical_events": summary["logical_events"],
        "materialized_event_objects": summary["materialized_event_objects"],
        "representation": summary["representation"],
        "exact_r_complete": summary["exact_r_complete"],
        "race_scan_scope": summary["race_scan_scope"],
        "race_ordering_performed": summary["race_ordering_performed"],
    }
    return record


def validate_successor(source_matrix: dict[str, Any], matrix: dict[str, Any], sync_gate: dict[str, Any], replay: dict[str, Any]) -> None:
    if matrix["supersedes"]["sha256"] != SOURCE_MATRIX_SHA256 or matrix["denominator"] != source_matrix["denominator"]:
        raise RuntimeError("source matrix identity or denominator changed")
    if len(matrix["records"]) != 6 or len(replay["records"]) != 6:
        raise RuntimeError("Suite 4 row count changed")
    source_by_id = {row["case_id"]: row for row in source_matrix["records"]}
    for record in matrix["records"]:
        case_id = record["case_id"]
        source = source_by_id[case_id]
        if record["baselines"] != source["baselines"] or record["audit"] != source["audit"]:
            raise RuntimeError(f"{case_id}: frozen audit/baselines changed")
        require_summary(case_id, record["pipeguard"], SUITE4_EXPECTED[case_id])
        if record["pipeguard"]["agreement"] is not True:
            raise RuntimeError(f"{case_id}: audit disagreement")
    if matrix["denominator"]["pipeguard_agreements"] != 6:
        raise RuntimeError("agreement denominator changed")
    if [record["point_id"] for record in sync_gate["records"]] != list(SYNC_EXPECTED):
        raise RuntimeError("SYNC point order changed")
    for record in sync_gate["records"]:
        require_summary(record["point_id"], record["summary"], SYNC_EXPECTED[record["point_id"]])


def producer_identity() -> dict[str, Any]:
    return {
        "branch": git("branch", "--show-current").decode().strip(),
        "head": git("rev-parse", "HEAD").decode().strip(),
        "tracked_dirty_diff_sha256": hashlib.sha256(git("diff", "--binary", "HEAD")).hexdigest(),
        "direct_core": ref(ROOT / "src/pipeguard/direct_parametric.py", "CHECKER_CORE"),
        "generator": ref(Path(__file__), "SUCCESSOR_REPLAY_GENERATOR"),
    }


def build(output_root: Path, matrix_output: Path) -> dict[str, Any]:
    validate_guards()
    if output_root.exists() or matrix_output.exists():
        raise FileExistsError(output_root if output_root.exists() else matrix_output)
    output_root.mkdir(parents=True, exist_ok=False)
    source_matrix = json.loads(SOURCE_MATRIX.read_text())
    sync_manifest = json.loads(SYNC_MANIFEST.read_text())
    producer = producer_identity()
    sync_records = []
    for point in sync_manifest["series"]["points"]:
        point_id = point["id"]
        if point_id not in SYNC_EXPECTED:
            raise RuntimeError(f"unexpected SYNC point {point_id}")
        pgir = resolve(point["pgir"])
        if sha256(pgir) != point["pgir_sha256"]:
            raise RuntimeError(f"{point_id}: PGIR digest mismatch")
        summary, artifacts = run_checker(pgir, f"sync-{point_id}", output_root)
        require_summary(point_id, summary, SYNC_EXPECTED[point_id])
        sync_records.append({"point_id": point_id, "case_id": point["case_id"], "summary": summary, "artifacts": artifacts})
    sync_gate = {"schema_version": "pipeguard-tl-core-22564-sync-gate-v1", "status": "SUCCESSOR_CORE_SYNC_GATE_COMPLETE", "producer": producer, "source_manifest": ref(SYNC_MANIFEST, "SOURCE_SERIES_MANIFEST"), "records": sync_records}
    sync_gate_path = output_root / "sync-gate.json"
    write_new(sync_gate_path, sync_gate)

    replay_records, successor_records = [], []
    for index, source_record in enumerate(source_matrix["records"]):
        case_id = source_record["case_id"]
        if case_id not in SUITE4_EXPECTED:
            raise RuntimeError(f"unexpected Suite 4 row {case_id}")
        pgir_entry = source_record["artifacts"]["pgir"]
        pgir = resolve(pgir_entry["path"])
        if sha256(pgir) != pgir_entry["sha256"]:
            raise RuntimeError(f"{case_id}: PGIR digest mismatch")
        summary, artifacts = run_checker(pgir, f"suite4-{index:02d}", output_root)
        require_summary(case_id, summary, SUITE4_EXPECTED[case_id])
        replay_records.append({"case_id": case_id, "summary": summary, "artifacts": artifacts})
        successor_records.append(successor_record(source_record, summary, artifacts["checker"]))
    replay = {
        "schema_version": "pipeguard-suite4-successor-core-replay-v1", "status": "SUCCESSOR_CORE_REPLAY_COMPLETE",
        "producer": producer, "source_matrix": ref(SOURCE_MATRIX, "PREDECESSOR_FINAL_CANDIDATE"), "records": replay_records,
        "summary": {"rows": 6, "agreements": sum(r["pipeguard"]["agreement"] for r in successor_records), "hazards": sum(r["pipeguard"]["verdict"] == "DATA_HAZARD" for r in successor_records), "safe_controls": sum(r["pipeguard"]["verdict"] == "SAFE" for r in successor_records)},
    }
    replay_path = output_root / "suite4-replay.json"
    write_new(replay_path, replay)

    matrix = copy.deepcopy(source_matrix)
    matrix["schema_version"] = "pipeguard-suite4-final-candidate-matrix-successor-v1"
    matrix["supersedes"] = ref(SOURCE_MATRIX, "PREDECESSOR_FINAL_CANDIDATE")
    matrix["successor_core_replay"] = ref(replay_path, "SUCCESSOR_CORE_REPLAY")
    matrix["successor_sync_gate"] = ref(sync_gate_path, "SUCCESSOR_CORE_SYNC_GATE")
    matrix["producer"] = producer
    matrix["records"] = successor_records
    matrix["integrity_notes"] = list(matrix["integrity_notes"]) + [
        "This successor replays only PipeGuard under core 22564a5d; frozen audits, numerical observations, sanitizer baselines, catalog membership, and denominator are unchanged.",
        "The predecessor 35df5f3e matrix remains immutable and is referenced rather than relabeled.",
    ]
    validate_successor(source_matrix, matrix, sync_gate, replay)
    write_new(matrix_output, matrix)
    acceptance = {
        "schema_version": "pipeguard-tl-core-22564-consumer-acceptance-v1", "status": "CONSUMER_ACCEPTANCE_COMPLETE",
        "producer": producer, "handoff_manifest_sha256": "5a8bdabc350d825cacae256102cd1bb355d8405fcfbac40e7f48ae0294f488e1",
        "sync_gate": ref(sync_gate_path), "suite4_replay": ref(replay_path), "successor_matrix": ref(matrix_output),
        "historical_guards": [{"path": p, "sha256": s} for p, s in GUARDS.items()],
    }
    write_new(output_root / "acceptance.json", acceptance)
    return acceptance


def check(output_root: Path, matrix_output: Path) -> dict[str, Any]:
    validate_guards()
    source_matrix = json.loads(SOURCE_MATRIX.read_text())
    matrix = json.loads(matrix_output.read_text())
    sync_path, replay_path = output_root / "sync-gate.json", output_root / "suite4-replay.json"
    sync_gate, replay = json.loads(sync_path.read_text()), json.loads(replay_path.read_text())
    acceptance = json.loads((output_root / "acceptance.json").read_text())
    validate_successor(source_matrix, matrix, sync_gate, replay)
    for key, path in (("sync_gate", sync_path), ("suite4_replay", replay_path), ("successor_matrix", matrix_output)):
        if acceptance[key]["sha256"] != sha256(path):
            raise RuntimeError(f"acceptance digest mismatch for {key}")
    for container in (sync_gate, replay):
        for record in container["records"]:
            for artifact in record["artifacts"].values():
                path = resolve(artifact["path"])
                if sha256(path) != artifact["sha256"]:
                    raise RuntimeError(f"artifact digest mismatch: {path}")
    return acceptance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--matrix-output", type=Path, default=MATRIX_OUTPUT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    result = check(args.output_root, args.matrix_output) if args.check else build(args.output_root, args.matrix_output)
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
