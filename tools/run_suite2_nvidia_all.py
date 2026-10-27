#!/usr/bin/env python3
"""Run NVIDIA Compute Sanitizer baselines over the full selected-210 suite.

Every manifest row is executed under both racecheck and synccheck so the
ledger captures the complete detection matrix: detection rate on hazard
rows, false-positive rate on SAFE rows, and timeout behaviour (deadlocked
kernels destroy the observation channel). Resume is supported: a case/mode
directory with a result.json is skipped.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from run_suite2_selected210 import normalize_case  # noqa: E402

MANIFEST = ROOT / "suite/suite2/manifest.json"
WORKER = ROOT / "tools/suite2_nvidia_worker.py"
PYTHONS = {
    "tilelang": Path(os.environ.get("PIPEGUARD_TL_PYTHON", "/home/xxw/PipeGuard-sort/envs/tl-pg/bin/python")),
    "triton": Path(os.environ.get("PIPEGUARD_TRITON_PYTHON", "/home/xxw/PipeGuard-sort/envs/triton-pg/bin/python")),
}
SANITIZER = Path(os.environ.get("PIPEGUARD_SANITIZER", "/usr/local/cuda-12.9/bin/compute-sanitizer"))
MODES = ("racecheck", "synccheck")


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


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
    if mode == "synccheck":
        if "Barrier error detected." in output:
            return "SYNCHECK_BARRIER_ERROR"
        if re.search(r"SYNCCHECK SUMMARY:\s*([1-9][0-9]*) errors? displayed", output):
            return "SYNCCHECK_ERROR"
    return None


def invoke(row: dict[str, Any], case_index: int, mode: str, gpu: int, output: Path, timeout: float) -> dict[str, Any]:
    case_id = row["case_id"]
    case_dir = output / "cases" / case_id / mode
    result_path = case_dir / "result.json"
    if result_path.is_file():
        return json.loads(result_path.read_text())
    worker_dir = case_dir / "worker"
    worker_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "home").mkdir(exist_ok=True)
    (case_dir / "triton-cache").mkdir(exist_ok=True)
    spec = normalize_case(row, case_index)
    if spec.get("standalone_source"):
        # The standalone Barrier/Memory sources are fixed 256-element kernels;
        # normalize_case's capture-only knobs violate the launch ABI.
        spec["semantic_knobs"].update({"elements": 255, "threads": 128, "stages": 2})
    spec_path = case_dir / "normalized-spec.json"
    spec_path.write_bytes(canonical(spec))
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
    hazard_expected = row["result_class"] != "SAFE"
    if concrete:
        effectiveness = "DETECTED" if hazard_expected else "FALSE_POSITIVE"
    elif timed_out:
        effectiveness = "TARGET_TIMEOUT"
    elif worker is not None and worker.get("execution_status") == "EXECUTED":
        effectiveness = "NO_DIAGNOSTIC" if hazard_expected else "CORRECT_NO_DIAGNOSTIC"
    elif worker is not None:
        effectiveness = "TARGET_ERROR"
    else:
        effectiveness = "SETUP_ERROR"
    record = {
        "case_id": case_id, "dsl": row["dsl"], "result_class": row["result_class"],
        "primitive": row.get("primitive"), "variant": row.get("variant"),
        "scenario_id": row.get("scenario_id"),
        "nvidia_applicability": row.get("nvidia_applicability"),
        "gpu": gpu, "mode": mode,
        "timeout_seconds": timeout, "elapsed_seconds": elapsed, "timed_out": timed_out,
        "returncode": process.returncode,
        "terminal_status": "TARGET_TIMEOUT" if timed_out else "COMPLETED",
        "effectiveness_status": effectiveness, "diagnostic": concrete, "worker": worker,
        "source": row["source"], "spec": row["spec"],
        "stdout": {"path": "stdout", "bytes": len(stdout), "sha256": sha_bytes(stdout)},
        "stderr": {"path": "stderr", "bytes": len(stderr), "sha256": sha_bytes(stderr)},
    }
    result_path.write_bytes(canonical(record))
    print(json.dumps({"case_id": case_id, "gpu": gpu, "mode": mode, "effectiveness_status": effectiveness, "diagnostic": concrete, "elapsed_seconds": round(elapsed, 3)}, sort_keys=True), flush=True)
    return record


def run_shard(items: list[tuple[dict[str, Any], int, str]], gpu: int, output: Path, timeout: float) -> list[dict[str, Any]]:
    return [invoke(row, case_index, mode, gpu, output, timeout) for row, case_index, mode in items]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--case-id", action="append")
    parser.add_argument("--mode", action="append", choices=MODES)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST.read_text())
    rows = manifest["cases"]
    if len(rows) != 210:
        raise RuntimeError(f"expected 210 manifest rows, got {len(rows)}")
    modes = tuple(args.mode) if args.mode else MODES
    selected = [
        (row, case_index, mode)
        for case_index, row in enumerate(rows)
        if not args.case_id or row["case_id"] in set(args.case_id)
        for mode in modes
    ]
    gpus = [int(value) for value in args.gpus.split(",")]
    shards: list[list[tuple[dict[str, Any], int, str]]] = [[] for _ in gpus]
    for index, item in enumerate(selected):
        shards[index % len(gpus)].append(item)
    records: list[dict[str, Any]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_shard, shard, gpu, output, args.timeout) for shard, gpu in zip(shards, gpus)]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda record: (record["case_id"], record["mode"]))
    ledger_path = output / "ledger.json"
    ledger_path.write_bytes(canonical({"schema_version": "pipeguard-suite2-nvidia-all-v1", "records": records}))

    def matrix(mode: str, result_class: str) -> dict[str, int]:
        return dict(sorted(Counter(
            record["effectiveness_status"]
            for record in records
            if record["mode"] == mode and record["result_class"] == result_class
        ).items()))

    summary = {
        "schema_version": "pipeguard-suite2-nvidia-all-summary-v1",
        "rows": len(records), "timeout_seconds": args.timeout, "modes": list(modes),
        "matrix": {
            mode: {result_class: matrix(mode, result_class) for result_class in ("SAFE", "DATA_HAZARD", "SYNC_HAZARD")}
            for mode in modes
        },
        "ledger": {"path": str(ledger_path), "bytes": ledger_path.stat().st_size, "sha256": sha_bytes(ledger_path.read_bytes())},
    }
    (output / "summary.json").write_bytes(canonical(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
