from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
BUILDER_PATH = ROOT / "tools/build_suite2_executable_matrix.py"
WORKER = ROOT / "tools/suite2_executable_worker.py"
PYTHONS = {
    "tilelang": Path("/usr/local/miniconda3/envs/tl-pg/bin/python"),
    "triton": Path("/usr/local/miniconda3/envs/triton-pg/bin/python"),
}
SANITIZER = Path("/usr/local/cuda-12.9/bin/compute-sanitizer")
TRITON_OPT = Path(
    "/mnt/project_dft/xuwen/PipeGuard-asplos/triton/"
    "build/cmake.linux-x86_64-cpython-3.10/bin/triton-opt"
)
MODE_BY_CLASS = {
    "SAFE": "synccheck",
    "DATA_HAZARD": "racecheck",
    "SYNC_HAZARD": "synccheck",
}
MODES = ("racecheck", "synccheck")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def write_once(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def load_builder():
    spec = importlib.util.spec_from_file_location("suite2_matrix_batch", BUILDER_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(BUILDER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def select_cases(specs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected = [
        spec for spec in specs
        if spec["primitive"] == "mbarrier" and 1 <= spec["variant"] <= 14
    ]
    selected.sort(key=lambda spec: spec["case_id"])
    if len(selected) != 84:
        raise RuntimeError(f"expected 84 remaining mbarrier rows, got {len(selected)}")
    return selected


def gpu_processes(index: int) -> list[str]:
    gpu_lines = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader,nounits"],
        text=True,
    ).splitlines()
    uuid = None
    for line in gpu_lines:
        candidate, value = [item.strip() for item in line.split(",", 1)]
        if candidate == str(index):
            uuid = value
            break
    if uuid is None:
        raise RuntimeError(f"GPU {index} not found")
    rows = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,gpu_uuid,process_name",
         "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return [row for row in rows if uuid in row]


def wait_for_gpu(index: int, poll_seconds: float) -> None:
    while gpu_processes(index):
        time.sleep(poll_seconds)


def worker_payload(stdout: bytes) -> dict[str, Any] | None:
    for line in reversed(stdout.decode(errors="replace").splitlines()):
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict) and "execution_status" in value:
            return value
    return None


def concrete_diagnostic(mode: str, output: str) -> str | None:
    if mode == "synccheck" and "Barrier error detected." in output:
        return "BARRIER_ERROR"
    if mode == "racecheck" and re.search(
        r"RACECHECK SUMMARY:\s*([1-9][0-9]*) hazards? displayed", output
    ):
        return "RACECHECK_HAZARD"
    if mode == "memcheck" and any(marker in output for marker in (
        "Invalid __global__", "Invalid __shared__", "Misaligned address",
        "Out-of-range Shared or Local Address",
    )):
        return "MEMCHECK_ACCESS_ERROR"
    if mode == "initcheck" and any(marker in output for marker in (
        "Uninitialized __global__ memory read", "Uninitialized access",
    )):
        return "INITCHECK_UNINITIALIZED_ACCESS"
    return None


def classify_invocation(
    *, mode: str, oracle: str, timed_out: bool, returncode: int | None,
    worker: dict[str, Any] | None, diagnostic: str | None,
    lowering_retained: bool,
) -> tuple[str, str]:
    if diagnostic is not None:
        return "DETECTED_TARGET_ABORTED" if worker and worker.get("execution_status") == "ERROR" else "DETECTED", "DETECTED"
    if timed_out:
        if lowering_retained:
            effect = "FAILED_TO_DETECT" if oracle != "SAFE" and mode != "native" else "NOT_APPLICABLE"
            return "TARGET_TIMEOUT", effect
        return "COMPILE_OR_SETUP_TIMEOUT", "NOT_SCORED"
    if worker is None:
        return "WRAPPER_ERROR", "NOT_SCORED"
    if worker.get("execution_status") == "EXECUTED":
        return "COMPLETED", "CORRECT_NO_DIAGNOSTIC" if oracle == "SAFE" else "NO_DIAGNOSTIC"
    if returncode not in (0, None):
        return "TOOL_ERROR", "NOT_SCORED"
    return "TARGET_ERROR", "NOT_SCORED"


def invoke(spec: dict[str, Any], mode: str, gpu: int, output: Path, timeout: float) -> dict[str, Any]:
    case_root = output / "cases" / spec["case_id"]
    prior_dirs = [case_root / mode, *sorted(case_root.glob(f"{mode}-retry-*"))]
    retryable = {
        "TARGET_ERROR", "RUNNER_ERROR", "WRAPPER_ERROR", "TOOL_ERROR",
        "COMPILE_OR_SETUP_TIMEOUT",
    }
    for prior_dir in prior_dirs:
        record_path = prior_dir / "invocation.json"
        if record_path.exists():
            prior = json.loads(record_path.read_text())
            if prior.get("terminal_status") not in retryable:
                stale_tma_capture = (
                    spec["dsl"] == "triton"
                    and spec["result_class"] == "DATA_HAZARD"
                    and (prior.get("worker") or {}).get("tma_descriptor_shape")
                    != [(spec["variant"] + 1) * 32, 32]
                )
                if not stale_tma_capture:
                    return prior
    case_dir = case_root / mode
    if case_dir.exists():
        retry = 1
        while (case_root / f"{mode}-retry-{retry}").exists():
            retry += 1
        case_dir = case_root / f"{mode}-retry-{retry}"
    record_path = case_dir / "invocation.json"
    case_dir.mkdir(parents=True, exist_ok=False)
    worker_dir = case_dir / "worker"
    worker_dir.mkdir()
    (case_dir / "home").mkdir()
    (case_dir / "triton-cache").mkdir()
    command = [
        str(PYTHONS[spec["dsl"]]), str(WORKER),
        "--spec", str(ROOT / spec["spec_path"]), "--output", str(worker_dir),
    ]
    if mode != "native":
        command = [str(SANITIZER), "--tool", mode, "--target-processes", "all", *command]
    env = dict(os.environ)
    env.update({
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "HOME": str(case_dir / "home"),
        "TRITON_CACHE_DIR": str(case_dir / "triton-cache"),
        "TRITON_PTXAS_PATH": "/usr/local/cuda-12.9/bin/ptxas",
        "PYTHONDONTWRITEBYTECODE": "1", "PYTHONNOUSERSITE": "1", "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "tests"), str(ROOT / "src"), str(ROOT / "tools"))),
    })
    wait_for_gpu(gpu, 10.0)
    started = time.monotonic()
    process = subprocess.Popen(
        command, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        start_new_session=True,
    )
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
    elapsed = time.monotonic() - started
    stdout_ref = write_once(case_dir / "stdout", stdout)
    stderr_ref = write_once(case_dir / "stderr", stderr)
    payload = worker_payload(stdout)
    combined = (stdout + b"\n" + stderr).decode(errors="replace")
    diagnostic = None if mode == "native" else concrete_diagnostic(mode, combined)
    lowering_names = (
        ("kernel.cu", "kernel.so") if spec["dsl"] == "tilelang"
        else ("raw.ttgir.mlir", "kernel.ptx", "kernel.cubin")
    )
    retained = all((worker_dir / name).exists() for name in lowering_names)
    terminal, effectiveness = classify_invocation(
        mode=mode, oracle=spec["result_class"], timed_out=timed_out,
        returncode=process.returncode, worker=payload, diagnostic=diagnostic,
        lowering_retained=retained,
    )
    artifacts = []
    for path in sorted(worker_dir.iterdir()):
        if path.is_file():
            artifacts.append({
                "path": str(path.relative_to(output)), "sha256": sha(path),
                "bytes": path.stat().st_size,
            })
    record = {
        "case_id": spec["case_id"], "dsl": spec["dsl"],
        "result_class": spec["result_class"], "variant": spec["variant"],
        "mode": mode, "gpu": gpu, "timeout_seconds": timeout,
        "elapsed_seconds": elapsed, "timed_out": timed_out,
        "returncode": process.returncode, "terminal_status": terminal,
        "effectiveness_status": effectiveness, "diagnostic": diagnostic,
        "worker": payload, "lowering_and_executable_retained": retained,
        "stdout": stdout_ref, "stderr": stderr_ref, "artifacts": artifacts,
    }
    write_once(record_path, canonical(record))
    return record


def verify_triton(spec: dict[str, Any], output: Path) -> dict[str, Any]:
    case_dir = output / "cases" / spec["case_id"]
    candidates = [
        path
        for mode in MODES
        for path in sorted(case_dir.glob(f"{mode}*/worker/raw.ttgir.mlir"))
    ]
    source = candidates[-1] if candidates else None
    prior_paths = sorted(case_dir.glob("triton-verifier*.json"))
    if source is not None:
        source_sha = sha(source)
        for prior_path in prior_paths:
            prior = json.loads(prior_path.read_text())
            if prior.get("input", {}).get("sha256") == source_sha:
                return prior
    if source is None:
        record = {"status": "NOT_RUN_NO_TTGIR", "structural_only": True}
    else:
        process = subprocess.run([str(TRITON_OPT), str(source)], capture_output=True, timeout=60)
        suffix = "" if not prior_paths else f"-retry-{len(prior_paths)}"
        stdout_ref = write_once(case_dir / f"triton-verifier{suffix}.stdout", process.stdout)
        stderr_ref = write_once(case_dir / f"triton-verifier{suffix}.stderr", process.stderr)
        record = {
            "status": "ACCEPTED_STRUCTURAL_ONLY" if process.returncode == 0 else "REJECTED_STRUCTURAL_ONLY",
            "returncode": process.returncode, "structural_only": True,
            "genuine_oracle_mismatch": False,
            "input": {"path": str(source.relative_to(output)), "sha256": sha(source)},
            "stdout": stdout_ref, "stderr": stderr_ref,
        }
    suffix = "" if not prior_paths else f"-retry-{len(prior_paths)}"
    path = case_dir / f"triton-verifier{suffix}.json"
    write_once(path, canonical(record))
    return record


def run_gpu(gpu: int, specs: list[dict[str, Any]], output: Path, timeout: float) -> list[dict[str, Any]]:
    records = []
    for spec in specs:
        invocations = []
        mode = MODE_BY_CLASS[spec["result_class"]]
        try:
            record = invoke(spec, mode, gpu, output, timeout)
        except Exception as error:
            record = {
                "case_id": spec["case_id"], "dsl": spec["dsl"],
                "result_class": spec["result_class"], "variant": spec["variant"],
                "mode": mode, "terminal_status": "RUNNER_ERROR",
                "effectiveness_status": "NOT_SCORED",
                "diagnostic": type(error).__name__, "message": str(error),
            }
            error_path = output / "cases" / spec["case_id"] / mode / "runner-error.json"
            if not error_path.exists():
                write_once(error_path, canonical(record))
        invocations.append(record)
        print(json.dumps({
            "case_id": spec["case_id"], "mode": mode,
            "terminal_status": record["terminal_status"], "gpu": gpu,
        }, sort_keys=True), flush=True)
        verifier = verify_triton(spec, output) if spec["dsl"] == "triton" else {"status": "NOT_APPLICABLE"}
        actual_knobs = dict(spec["semantic_knobs"])
        if spec["dsl"] == "triton":
            launch = (record.get("worker") or {}).get("launch") or {}
            if "num_warps" in launch:
                actual_knobs["triton_num_warps"] = launch["num_warps"]
                actual_knobs["triton_launch_threads"] = launch["num_warps"] * 32
            if spec["result_class"] == "DATA_HAZARD":
                actual_knobs["triton_tma_block"] = 32
                actual_knobs["triton_tma_transfers"] = spec["variant"] + 1
                actual_knobs["triton_tma_descriptor_shape"] = [
                    (spec["variant"] + 1) * 32,
                    32,
                ]
        case_record = {
            "case_id": spec["case_id"], "dsl": spec["dsl"],
            "result_class": spec["result_class"], "variant": spec["variant"],
            "source": spec["source"], "spec_path": spec["spec_path"],
            "semantic_knobs": actual_knobs,
            "invocations": invocations, "triton_verifier": verifier,
            "ai_adjudication": {"status": "PENDING_BATCH_REVIEW"},
        }
        case_root = output / "cases" / spec["case_id"]
        case_path = case_root / "case.json"
        if case_path.exists():
            retry = 1
            while (case_root / f"case-retry-{retry}.json").exists():
                retry += 1
            case_path = case_root / f"case-retry-{retry}.json"
        write_once(case_path, canonical(case_record))
        records.append(case_record)
    return records


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--timeout", type=float, default=120.0)
    args = parser.parse_args(argv)
    corpus = args.corpus.resolve(); output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected_path = output / "selected.json"
    if selected_path.exists():
        selected = select_cases(json.loads(selected_path.read_text())["cases"])
    else:
        builder = load_builder(); specs = builder.build_specs(corpus)
        manifest = {"schema_version": "pipeguard-suite2-mbarrier-batch-v1", "cases": specs}
        manifest_path = corpus / "manifest.json"
        if not manifest_path.exists():
            builder.write_once(manifest_path, canonical(manifest))
        selected = select_cases(specs)
        selected_manifest = {
            "schema_version": "pipeguard-suite2-mbarrier-batch-v1",
            "status": "RUNNING", "expected_rows": 84,
            "variants": list(range(1, 15)),
            "mode_by_result_class": MODE_BY_CLASS,
            "timeout_seconds": args.timeout, "cases": selected,
        }
        write_once(selected_path, canonical(selected_manifest))
    gpus = [int(value) for value in args.gpus.split(",") if value]
    shards = {gpu: [] for gpu in gpus}
    for index, spec in enumerate(selected):
        shards[gpus[index % len(gpus)]].append(spec)
    records = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        futures = [executor.submit(run_gpu, gpu, shards[gpu], output, args.timeout) for gpu in gpus]
        for future in concurrent.futures.as_completed(futures):
            records.extend(future.result())
    records.sort(key=lambda record: record["case_id"])
    terminal = Counter(
        invocation["terminal_status"]
        for record in records for invocation in record["invocations"]
    )
    summary = {
        "schema_version": "pipeguard-suite2-mbarrier-batch-v1",
        "status": "EXECUTION_COMPLETE_PENDING_INDEPENDENT_BATCH_REVIEW",
        "expected_rows": 84, "rows": len(records),
        "invocations_expected": 84,
        "invocations": sum(len(record["invocations"]) for record in records),
        "terminal_statuses": dict(sorted(terminal.items())),
        "ai_adjudication_pending": len(records),
        "pipeguard_run": False,
    }
    write_once(output / "ledger.json", canonical({"records": records}))
    write_once(output / "summary.json", canonical(summary))
    print(json.dumps(summary, sort_keys=True))
    return 0 if len(records) == 84 else 2


if __name__ == "__main__":
    raise SystemExit(main())
