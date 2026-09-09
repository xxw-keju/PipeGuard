from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests")]

from pipeguard import check_module
from pipeguard.frontends.tilelang import tilelang_snapshot_to_module
from pipeguard.frontends.triton import triton_snapshot_to_module

MANIFEST = ROOT / "suite/suite2/manifest.json"
WORKER = ROOT / "tools/suite2_selected_worker.py"
PYTHONS = {
    "tilelang": Path("/usr/local/miniconda3/envs/tl-pg/bin/python"),
    "triton": Path("/usr/local/miniconda3/envs/triton-pg/bin/python"),
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def artifact(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha(path)}


def normalize_case(row: dict[str, Any], case_index: int) -> dict[str, Any]:
    source_path = ROOT / row["source"]["path"]
    spec_path = ROOT / row["spec"]["path"]
    if sha(source_path) != row["source"]["sha256"]:
        raise RuntimeError(f"source hash mismatch: {source_path}")
    if sha(spec_path) != row["spec"]["sha256"]:
        raise RuntimeError(f"spec hash mismatch: {spec_path}")
    original = json.loads(spec_path.read_text())
    if "template_family" in original:
        spec = dict(original)
        spec["standalone_source"] = False
    else:
        primitive = row["primitive"]
        spec = {
            "case_id": row["case_id"],
            "dsl": row["dsl"],
            "result_class": row["result_class"],
            "primitive": primitive,
            "variant": row["variant"],
            "template_family": "CTA-HANDOFF" if primitive == "Barrier" else "MEM-RAW",
            "scenario_id": primitive.upper(),
            "polarity": "CORRECT" if row["result_class"] == "SAFE" else "BUG",
            "semantic_knobs": {
                "elements": row["variant"] + 1,
                "stages": 2,
                "threads": 128,
                "region_offset_bytes": 0,
                "protocol_depth": 1,
                "tilelang_tma_bytes": 16,
                "triton_tma_block": 32,
                "triton_tma_transfers": 1,
                "wgmma_repeats": 1,
                "wgmma_shape": [64, 64, 16],
                "proxy_repeats": 1,
                "proxy_store_shape": [64, 64],
            },
            "oracle": {
                "verdict": row["result_class"],
                "basis": "independent Suite 2 Barrier/Memory source label",
            },
            "ai_adjudication": row.get("adjudication", {}),
            "admitted": False,
            "standalone_source": True,
        }
    spec["source"] = dict(row["source"])
    spec["selected_manifest_row"] = case_index
    spec["gpu"] = case_index % 4
    return spec


def environment(case_dir: Path, gpu: int) -> dict[str, str]:
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HOME": str(case_dir / "home"),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join((str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests"))),
        "TRITON_CACHE_DIR": str(case_dir / "triton-cache"),
        "TRITON_PTXAS_PATH": "/usr/local/cuda-12.9/bin/ptxas",
    })
    return env


def run_case(spec: dict[str, Any], row: dict[str, Any], output: Path, timeout: float) -> dict[str, Any]:
    case_dir = output / "cases" / spec["case_id"]
    result_path = case_dir / "result.json"
    if result_path.exists():
        return json.loads(result_path.read_text())
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "home").mkdir(exist_ok=True)
    (case_dir / "triton-cache").mkdir(exist_ok=True)
    normalized_spec = case_dir / "normalized-spec.json"
    atomic_json(normalized_spec, spec)
    command = [str(PYTHONS[spec["dsl"]]), str(WORKER), "--spec", str(normalized_spec), "--output", str(case_dir)]
    started = time.monotonic()
    try:
        process = subprocess.run(
            command,
            cwd=ROOT,
            env=environment(case_dir, spec["gpu"]),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        (case_dir / "worker.stdout").write_text(process.stdout)
        (case_dir / "worker.stderr").write_text(process.stderr)
        if process.returncode == 0 and process.stdout.splitlines():
            worker = json.loads(process.stdout.splitlines()[-1])
        else:
            worker = {
                "execution_status": "ERROR",
                "capture_status": "ERROR",
                "diagnostic": "WORKER_EXIT",
                "returncode": process.returncode,
            }
    except subprocess.TimeoutExpired as error:
        stdout = error.stdout or ""
        stderr = error.stderr or ""
        if isinstance(stdout, bytes):
            stdout = stdout.decode(errors="replace")
        if isinstance(stderr, bytes):
            stderr = stderr.decode(errors="replace")
        (case_dir / "worker.stdout").write_text(stdout)
        (case_dir / "worker.stderr").write_text(stderr)
        worker = {"execution_status": "TIMEOUT", "capture_status": "TIMEOUT", "diagnostic": "WORKER_TIMEOUT"}

    report = None
    pgir_ref = None
    report_ref = None
    if worker.get("capture_status") == "CAPTURED":
        snapshot_path = case_dir / worker["snapshot"]["path"]
        try:
            snapshot = json.loads(snapshot_path.read_text())
            converter = tilelang_snapshot_to_module if spec["dsl"] == "tilelang" else triton_snapshot_to_module
            module = converter(snapshot)
            pgir_path = case_dir / "pgir.json"
            atomic_json(pgir_path, module.raw)
            report = check_module(module, engine="maximal")
            report_path = case_dir / "checker.json"
            atomic_json(report_path, report)
            pgir_ref = artifact(pgir_path)
            report_ref = artifact(report_path)
            terminal = "CHECKED"
        except Exception as error:
            terminal = "CHECK_FAILED"
            worker["check_error"] = f"{type(error).__name__}: {error}"
    elif worker.get("capture_status") == "UNSUPPORTED":
        terminal = "CAPTURE_UNSUPPORTED"
    elif worker.get("execution_status") == "TIMEOUT":
        terminal = "TIMEOUT"
    else:
        terminal = "COMPILE_FAILED"

    files = {
        name: artifact(case_dir / filename)
        for name, filename in {
            "raw_ttir": "raw.tvm.json",
            "raw_ttgir": "raw.ttgir.mlir",
            "final_cuda": "kernel.cu",
            "final_ptx": "kernel.ptx",
            "binary_so": "kernel.so",
            "binary_cubin": "kernel.cubin",
            "snapshot": "snapshot.json",
            "binding_evidence": "binding-evidence.json",
            "descriptor_evidence": "descriptor-evidence.json",
            "stdout": "worker.stdout",
            "stderr": "worker.stderr",
        }.items()
    }
    result = {
        "schema_version": "pipeguard-suite2-selected210-full-recapture-v1",
        "case_id": spec["case_id"],
        "dsl": spec["dsl"],
        "primitive": spec["primitive"],
        "result_class": spec["result_class"],
        "variant": spec["variant"],
        "standalone_source": spec["standalone_source"],
        "source": spec["source"],
        "expected_verdict": spec["oracle"]["verdict"],
        "old_verdict": row["pipeguard"]["verdict"],
        "terminal_status": terminal,
        "capture_status": worker.get("capture_status"),
        "new_verdict": report.get("verdict") if report else None,
        "new_primary_diagnostic": report.get("diagnostics", [{}])[0].get("code") if report and report.get("diagnostics") else None,
        "matches_expected": bool(report and report.get("verdict") == spec["oracle"]["verdict"]),
        "verdict_changed": bool(report and report.get("verdict") != row["pipeguard"]["verdict"]),
        "gpu": spec["gpu"],
        "elapsed_seconds": time.monotonic() - started,
        "worker": worker,
        "files": files,
        "pgir": pgir_ref,
        "checker_report": report_ref,
        "statistics": report.get("statistics") if report else None,
    }
    atomic_json(result_path, result)
    return result


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    checked = [record for record in records if record["terminal_status"] == "CHECKED"]
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        groups[":".join((record["primitive"], record["dsl"], record["result_class"]))].append(record)
    return {
        "schema_version": "pipeguard-suite2-selected210-full-recapture-summary-v1",
        "status": "COMPLETE" if len(records) == 210 and all(record["terminal_status"] in {"CHECKED", "CAPTURE_UNSUPPORTED", "COMPILE_FAILED", "CHECK_FAILED", "TIMEOUT"} for record in records) else "PARTIAL",
        "rows": len(records),
        "terminal_statuses": dict(Counter(record["terminal_status"] for record in records)),
        "capture_statuses": dict(Counter(str(record["capture_status"]) for record in records)),
        "new_verdicts": dict(Counter(str(record["new_verdict"]) for record in checked)),
        "new_primary_diagnostics": dict(Counter(str(record["new_primary_diagnostic"]) for record in checked)),
        "checked": len(checked),
        "matches_expected": sum(record["matches_expected"] for record in checked),
        "mismatches_expected": sum(not record["matches_expected"] for record in checked),
        "verdict_changed": sum(record["verdict_changed"] for record in checked),
        "groups": {
            key: {
                "rows": len(rows),
                "terminal_statuses": dict(Counter(row["terminal_status"] for row in rows)),
                "new_verdicts": dict(Counter(str(row["new_verdict"]) for row in rows if row["terminal_status"] == "CHECKED")),
                "matches_expected": sum(row["matches_expected"] for row in rows),
            }
            for key, rows in sorted(groups.items())
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--timeout", type=float, default=90.0)
    parser.add_argument("--case-id", action="append")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["cases"]
    if len(rows) != 210 or len({row["case_id"] for row in rows}) != 210:
        raise RuntimeError("Suite 2 manifest must contain 210 unique cases")
    selected = [row for row in rows if not args.case_id or row["case_id"] in set(args.case_id)]
    normalized = [(normalize_case(row, index), row) for index, row in enumerate(rows) if row in selected]
    atomic_json(output / "selection.json", {
        "manifest": {"path": str(MANIFEST), "sha256": sha(MANIFEST)},
        "selected_case_ids": [spec["case_id"] for spec, _row in normalized],
    })
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as executor:
        futures = {executor.submit(run_case, spec, row, output, args.timeout): spec for spec, row in normalized}
        for index, future in enumerate(concurrent.futures.as_completed(futures), 1):
            record = future.result()
            records.append(record)
            records.sort(key=lambda item: item["case_id"])
            atomic_json(output / "ledger.partial.json", {"records": records})
            if index % 10 == 0 or index == len(normalized):
                counts = Counter(item["terminal_status"] for item in records)
                print(json.dumps({"completed": index, "total": len(normalized), "terminal_statuses": dict(counts)}, sort_keys=True), flush=True)
    summary = summarize(records)
    identities = {
        "base_head": subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip(),
        "dirty_worktree": True,
        "checker_sha256": sha(ROOT / "src/pipeguard/checker.py"),
        "direct_core_sha256": sha(ROOT / "src/pipeguard/direct_parametric.py"),
        "tilelang_capture_sha256": sha(ROOT / "tools/tilelang_capture.py"),
        "triton_capture_sha256": sha(ROOT / "tools/triton_capture.py"),
        "worker_sha256": sha(WORKER),
        "runner_sha256": sha(Path(__file__)),
        "tilelang_case_helper_sha256": sha(
            ROOT / "suite/frontend_pairs/sources/tilelang_cases.py"
        ),
        "triton_case_helper_sha256": sha(
            ROOT / "suite/frontend_pairs/sources/triton_cases.py"
        ),
    }
    summary["identities"] = identities
    ledger_path = output / "ledger.json"
    atomic_json(ledger_path, {"schema_version": "pipeguard-suite2-selected210-full-recapture-v1", "records": records})
    summary["artifacts"] = {"ledger": artifact(ledger_path), "selection": artifact(output / "selection.json")}
    atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
