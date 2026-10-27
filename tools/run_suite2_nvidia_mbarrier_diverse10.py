#!/usr/bin/env python3
"""Run SyncCheck on the ten selected diverse mbarrier SYNC_HAZARD cases."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "suite/suite2/manifest.json"
WORKER = ROOT / "tools/suite2_nvidia_worker.py"
SANITIZER = Path("/usr/local/cuda-12.9/bin/compute-sanitizer")
PYTHONS = {
    "tilelang": Path(os.environ.get("PIPEGUARD_TL_PYTHON", "/home/xxw/PipeGuard-sort/envs/tl-pg/bin/python")),
    "triton": Path(os.environ.get("PIPEGUARD_TRITON_PYTHON", "/home/xxw/PipeGuard-sort/envs/triton-pg/bin/python")),
}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def worker_payload(stdout: bytes) -> dict[str, Any] | None:
    for line in reversed(stdout.decode(errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "execution_status" in value:
            return value
    return None


def diagnostic(output: str) -> str | None:
    if "Barrier error detected. Missing wait." in output:
        return "BARRIER_ERROR_MISSING_WAIT"
    if "Barrier error detected." in output:
        return "BARRIER_ERROR"
    match = re.search(r"ERROR SUMMARY:\s*([1-9][0-9]*) errors?", output)
    if match:
        return "SYNCHECK_ERROR"
    return None


def invoke(spec_record: dict[str, Any], gpu: int, output: Path, timeout: float) -> dict[str, Any]:
    case_id = spec_record["case_id"]
    dsl = spec_record["dsl"]
    case_dir = output / "cases" / case_id
    worker_dir = case_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / "home").mkdir()
    (case_dir / "triton-cache").mkdir()
    spec_path = ROOT / spec_record["spec"]["path"]
    command = [
        str(SANITIZER), "--tool", "synccheck", "--target-processes", "all",
        str(PYTHONS[dsl]), str(WORKER), "--spec", str(spec_path), "--output", str(worker_dir),
    ]
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu),
        HOME=str(case_dir / "home"),
        TRITON_CACHE_DIR=str(case_dir / "triton-cache"),
        TRITON_PTXAS_PATH="/usr/local/cuda-12.9/bin/ptxas",
        PYTHONPATH=os.pathsep.join((str(ROOT / "tests"), str(ROOT / "src"), str(ROOT / "tools"))),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONNOUSERSITE="1",
        PYTHONHASHSEED="0",
    )
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
        timed_out = False
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=3)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    (case_dir / "stdout").write_bytes(stdout)
    (case_dir / "stderr").write_bytes(stderr)
    payload = worker_payload(stdout)
    combined = (stdout + b"\n" + stderr).decode(errors="replace")
    concrete = diagnostic(combined)
    if concrete:
        effectiveness = "DETECTED"
    elif timed_out:
        effectiveness = "TARGET_TIMEOUT"
    elif payload is not None and payload.get("execution_status") == "EXECUTED":
        effectiveness = "NO_DIAGNOSTIC"
    elif payload is not None:
        effectiveness = "TARGET_ERROR"
    else:
        effectiveness = "SETUP_ERROR"
    artifacts = []
    for path in sorted(worker_dir.iterdir()):
        if path.is_file():
            artifacts.append({"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)})
    record = {
        "case_id": case_id,
        "dsl": dsl,
        "primitive": "mbarrier",
        "result_class": spec_record["result_class"],
        "variant": spec_record["variant"],
        "scenario_id": spec_record["scenario_id"],
        "gpu": gpu,
        "mode": "synccheck",
        "timeout_seconds": timeout,
        "elapsed_seconds": elapsed,
        "timed_out": timed_out,
        "returncode": process.returncode,
        "diagnostic": concrete,
        "effectiveness_status": effectiveness,
        "worker": payload,
        "source": spec_record["source"],
        "spec": spec_record["spec"],
        "stdout": {"path": "stdout", "bytes": len(stdout), "sha256": sha_bytes(stdout)},
        "stderr": {"path": "stderr", "bytes": len(stderr), "sha256": sha_bytes(stderr)},
        "artifacts": artifacts,
    }
    (case_dir / "result.json").write_bytes(canonical(record))
    print(json.dumps({
        "case_id": case_id,
        "gpu": gpu,
        "scenario_id": record["scenario_id"],
        "effectiveness_status": effectiveness,
        "diagnostic": concrete,
        "elapsed_seconds": round(elapsed, 3),
    }, sort_keys=True), flush=True)
    return record


def run_shard(rows: list[dict[str, Any]], gpu: int, output: Path, timeout: float) -> list[dict[str, Any]]:
    return [invoke(row, gpu, output, timeout) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--result-class", choices=("SAFE", "SYNC_HAZARD"), default="SYNC_HAZARD")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text())
    rows = [
        row for row in manifest["cases"]
        if row.get("primitive") == "mbarrier"
        and row.get("result_class") == args.result_class
        and row.get("template_family") == "MBARRIER-DIVERSE"
    ]
    rows.sort(key=lambda row: row["case_id"])
    if len(rows) != 10:
        raise RuntimeError(f"expected ten diverse mbarrier {args.result_class} cases, got {len(rows)}")
    gpus = [int(value) for value in args.gpus.split(",")]
    shards = [[] for _ in gpus]
    for index, row in enumerate(rows):
        shards[index % len(gpus)].append(row)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [
            executor.submit(run_shard, shard, gpu, output, args.timeout)
            for shard, gpu in zip(shards, gpus)
        ]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: row["case_id"])
    ledger = {"schema_version": "pipeguard-suite2-mbarrier-diverse-synccheck-v1", "records": records}
    ledger_path = output / "ledger.json"
    ledger_path.write_bytes(canonical(ledger))
    summary = {
        "schema_version": "pipeguard-suite2-mbarrier-diverse-synccheck-summary-v1",
        "rows": len(records),
        "timeout_seconds": args.timeout,
        "effectiveness_statuses": dict(sorted(Counter(row["effectiveness_status"] for row in records).items())),
        "diagnostics": dict(sorted(Counter(str(row["diagnostic"]) for row in records).items())),
        "by_scenario": {
            f"{row['dsl']}:{row['scenario_id']}": {
                "case_id": row["case_id"],
                "dsl": row["dsl"],
                "effectiveness_status": row["effectiveness_status"],
                "diagnostic": row["diagnostic"],
                "timed_out": row["timed_out"],
            }
            for row in records
        },
        "ledger": {"path": str(ledger_path), "bytes": ledger_path.stat().st_size, "sha256": sha(ledger_path)},
    }
    summary_path = output / "summary.json"
    summary_path.write_bytes(canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
