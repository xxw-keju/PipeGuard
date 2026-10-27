#!/usr/bin/env python3
"""Run applicable NVIDIA baselines for the diverse Suite 2 Barrier pairs."""

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
PYTHONS = {
    "tilelang": Path(os.environ.get("PIPEGUARD_TL_PYTHON", "/home/xxw/PipeGuard-sort/envs/tl-pg/bin/python")),
    "triton": Path(os.environ.get("PIPEGUARD_TRITON_PYTHON", "/home/xxw/PipeGuard-sort/envs/triton-pg/bin/python")),
}
SANITIZER = Path("/usr/local/cuda-12.9/bin/compute-sanitizer")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def normalize(row: dict[str, Any], gpu: int) -> dict[str, Any]:
    spec = json.loads((ROOT / row["spec"]["path"]).read_text())
    return {
        "case_id": row["case_id"],
        "dsl": row["dsl"],
        "result_class": row["result_class"],
        "primitive": "Barrier",
        "variant": row["variant"],
        "template_family": "CTA-HANDOFF",
        "scenario_id": row["scenario_id"],
        "polarity": "CORRECT" if row["result_class"] == "SAFE" else "BUG",
        "semantic_knobs": {"elements": 255, "threads": 128, "stages": 2},
        "oracle": spec["oracle"],
        "source": row["source"],
        "standalone_source": True,
        "gpu": gpu,
    }


def payload(stdout: bytes) -> dict[str, Any] | None:
    for line in reversed(stdout.decode(errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "execution_status" in value:
            return value
    return None


def parse_diagnostic(mode: str, output: str) -> str | None:
    if mode == "racecheck" and re.search(r"RACECHECK SUMMARY:\s*([1-9][0-9]*) hazards? displayed", output):
        return "RACECHECK_HAZARD"
    if mode == "synccheck" and "Barrier error detected." in output:
        return "SYNCHECK_BARRIER_ERROR"
    return None


def invoke(row: dict[str, Any], gpu: int, output: Path, timeout: float) -> dict[str, Any]:
    case_id = row["case_id"]
    if row["result_class"] == "DATA_HAZARD" and row["nvidia_applicability"] != "APPLICABLE_SHARED_MEMORY":
        return {
            "case_id": case_id,
            "dsl": row["dsl"],
            "result_class": row["result_class"],
            "primitive": "Barrier",
            "variant": row["variant"],
            "scenario_id": row["scenario_id"],
            "nvidia_applicability": row["nvidia_applicability"],
            "mode": "racecheck",
            "terminal_status": "NOT_APPLICABLE",
            "effectiveness_status": "NOT_APPLICABLE",
            "diagnostic": None,
            "timed_out": False,
            "source": row["source"],
            "spec": row["spec"],
        }
    case_dir = output / "cases" / case_id
    worker_dir = case_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=False)
    (case_dir / "home").mkdir()
    (case_dir / "triton-cache").mkdir()
    spec = normalize(row, gpu)
    spec_path = case_dir / "normalized-spec.json"
    spec_path.write_bytes(canonical(spec))
    mode = "synccheck" if row["result_class"] == "SAFE" else "racecheck"
    command = [
        str(SANITIZER), "--tool", mode, "--target-processes", "all",
        str(PYTHONS[row["dsl"]]), str(WORKER), "--spec", str(spec_path), "--output", str(worker_dir),
    ]
    env = dict(os.environ)
    env.update(
        CUDA_VISIBLE_DEVICES=str(gpu), HOME=str(case_dir / "home"),
        TRITON_CACHE_DIR=str(case_dir / "triton-cache"),
        TRITON_PTXAS_PATH="/usr/local/cuda-12.9/bin/ptxas",
        PYTHONPATH=os.pathsep.join((str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests"))),
        PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1", PYTHONHASHSEED="0",
    )
    started = time.monotonic()
    process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
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
    worker = payload(stdout)
    concrete = parse_diagnostic(mode, (stdout + b"\n" + stderr).decode(errors="replace"))
    if concrete:
        effectiveness = "DETECTED" if row["result_class"] != "SAFE" else "FALSE_POSITIVE"
    elif timed_out:
        effectiveness = "TARGET_TIMEOUT"
    elif worker is not None and worker.get("execution_status") == "EXECUTED":
        effectiveness = "CORRECT_NO_DIAGNOSTIC" if row["result_class"] == "SAFE" else "NO_DIAGNOSTIC"
    elif worker is not None:
        effectiveness = "TARGET_ERROR"
    else:
        effectiveness = "SETUP_ERROR"
    artifacts = [
        {"path": path.name, "bytes": path.stat().st_size, "sha256": sha(path)}
        for path in sorted(worker_dir.iterdir()) if path.is_file()
    ]
    record = {
        "case_id": case_id, "dsl": row["dsl"], "result_class": row["result_class"],
        "primitive": "Barrier", "variant": row["variant"], "scenario_id": row["scenario_id"],
        "nvidia_applicability": row["nvidia_applicability"], "gpu": gpu, "mode": mode,
        "timeout_seconds": timeout, "elapsed_seconds": elapsed, "timed_out": timed_out,
        "returncode": process.returncode, "terminal_status": "COMPLETED" if not timed_out else "TARGET_TIMEOUT",
        "effectiveness_status": effectiveness, "diagnostic": concrete, "worker": worker,
        "source": row["source"], "spec": row["spec"],
        "stdout": {"path": "stdout", "bytes": len(stdout), "sha256": sha_bytes(stdout)},
        "stderr": {"path": "stderr", "bytes": len(stderr), "sha256": sha_bytes(stderr)},
        "artifacts": artifacts,
    }
    (case_dir / "result.json").write_bytes(canonical(record))
    print(json.dumps({"case_id": case_id, "gpu": gpu, "mode": mode, "effectiveness_status": effectiveness, "diagnostic": concrete, "elapsed_seconds": round(elapsed, 3)}, sort_keys=True), flush=True)
    return record


def run_shard(rows: list[dict[str, Any]], gpu: int, output: Path, timeout: float) -> list[dict[str, Any]]:
    return [invoke(row, gpu, output, timeout) for row in rows]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(MANIFEST.read_text())
    rows = [row for row in manifest["cases"] if row.get("primitive") == "Barrier" and row.get("result_class") in {"SAFE", "DATA_HAZARD"} and row.get("source_binding") == "BARRIER_DIVERSE_V1"]
    rows.sort(key=lambda row: row["case_id"])
    if len(rows) != 20:
        raise RuntimeError(f"expected 20 Barrier rows, got {len(rows)}")
    gpus = [int(value) for value in args.gpus.split(",")]
    applicable = [row for row in rows if not (row["result_class"] == "DATA_HAZARD" and row["nvidia_applicability"] != "APPLICABLE_SHARED_MEMORY")]
    not_applicable = [invoke(row, 0, output, args.timeout) for row in rows if row not in applicable]
    shards = [[] for _ in gpus]
    for index, row in enumerate(applicable):
        shards[index % len(gpus)].append(row)
    records = list(not_applicable)
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_shard, shard, gpu, output, args.timeout) for shard, gpu in zip(shards, gpus)]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda row: row["case_id"])
    ledger_path = output / "ledger.json"
    ledger_path.write_bytes(canonical({"schema_version": "pipeguard-suite2-barrier-diverse-nvidia-v1", "records": records}))
    summary = {
        "schema_version": "pipeguard-suite2-barrier-diverse-nvidia-summary-v1",
        "rows": len(records), "invocations": len(applicable), "timeout_seconds": args.timeout,
        "effectiveness_statuses": dict(sorted(Counter(row["effectiveness_status"] for row in records).items())),
        "diagnostics": dict(sorted(Counter(str(row["diagnostic"]) for row in records).items())),
        "ledger": {"path": str(ledger_path), "bytes": ledger_path.stat().st_size, "sha256": sha(ledger_path)},
    }
    (output / "summary.json").write_bytes(canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
