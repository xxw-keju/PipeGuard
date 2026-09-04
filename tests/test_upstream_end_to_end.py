from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parents[1]
TILELANG_ROOT = Path(os.environ.get("TILELANG_ROOT", "/mnt/project_dft/xuwen/PipeGuard-asplos/tilelang"))
TRITON_ROOT = Path(os.environ.get("TRITON_ROOT", "/mnt/project_dft/xuwen/PipeGuard-asplos/triton"))
TRITON_PTXAS = Path(os.environ.get("TRITON_PTXAS_PATH", "/usr/local/cuda-12.9/bin/ptxas"))


def test_frontend_import_does_not_load_checker():
    process = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; import pipeguard.frontends; assert 'pipeguard.checker' not in sys.modules",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.skipif(
    not TILELANG_ROOT.exists() or not TRITON_ROOT.exists() or not TRITON_PTXAS.exists(),
    reason="pinned upstream compiler inputs unavailable",
)
def test_native_upstream_capture_parses_pgir_without_checker(tmp_path: Path):
    command = [
        sys.executable,
        str(ROOT / "tools" / "upstream_run.py"),
        "--tilelang-root", str(TILELANG_ROOT),
        "--triton-root", str(TRITON_ROOT),
        "--triton-ptxas", str(TRITON_PTXAS),
        "--candidate", "3ed245d0c48fd3d1",
        "--candidate", "24cf86a8b5051c08",
        "--output", str(tmp_path / "run"),
        "--jobs", "2",
        "--progress-interval", "0.05",
    ]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["checker_loaded"] is False
    assert summary["tilelang"] == {"CAPTURED": 1}
    assert summary["triton"] == {"CAPTURED": 1}
    run_root = tmp_path / "run" / summary["run_id"]
    result_paths = list(run_root.glob("*/*/*/result.json"))
    results = [json.loads(path.read_text()) for path in result_paths]
    assert len(results) == 2
    assert all(result["status"] == "CAPTURED" for result in results)
    assert all(result["checker_loaded"] is False for result in results)
    assert all(any(key.startswith("pgir:") for key in result["artifacts"]) for result in results)
    for result_path, result in zip(result_paths, results):
        for key, artifact in result["artifacts"].items():
            if not key.startswith("pgir:"):
                continue
            metadata_path = result_path.parent / artifact["metadata_path"]
            assert hashlib.sha256(metadata_path.read_bytes()).hexdigest() == artifact["metadata_sha256"]
            assert json.loads(metadata_path.read_text())["checker_loaded"] is False
    progress = [json.loads(line) for line in (run_root / "progress.jsonl").read_text().splitlines()]
    assert progress[0]["selected_total"] == 2
    assert progress[-1]["completed"] == 2
    corrupted_result = results[0]
    corrupted_artifact = next(
        artifact for key, artifact in corrupted_result["artifacts"].items() if key.startswith("pgir:")
    )
    corrupted_path = result_paths[0].parent / corrupted_artifact["metadata_path"]
    corrupted_path.write_text("{}\n")
    resumed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    assert resumed.returncode == 0, resumed.stderr
    resumed_summary = json.loads(resumed.stdout)
    assert resumed_summary["resumed"] == 1
    assert (
        hashlib.sha256(corrupted_path.read_bytes()).hexdigest()
        == corrupted_artifact["metadata_sha256"]
    )
    assert json.loads(corrupted_path.read_text())["checker_loaded"] is False
    fully_resumed = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=60)
    assert fully_resumed.returncode == 0, fully_resumed.stderr
    assert json.loads(fully_resumed.stdout)["resumed"] == 2


@pytest.mark.skipif(not TILELANG_ROOT.exists() or not TRITON_ROOT.exists(), reason="pinned upstream compiler inputs unavailable")
def test_pgir_parse_timeout_is_process_isolated(tmp_path: Path):
    parser = tmp_path / "slow-parser.py"
    parser.write_text("import time\ntime.sleep(60)\n")
    command = [
        sys.executable,
        str(ROOT / "tools" / "upstream_run.py"),
        "--dsl", "tilelang",
        "--tilelang-root", str(TILELANG_ROOT),
        "--triton-root", str(TRITON_ROOT),
        "--candidate", "3ed245d0c48fd3d1",
        "--parser", str(parser),
        "--parse-timeout", "0.1",
        "--output", str(tmp_path / "timeout-run"),
        "--progress-interval", "10",
    ]
    process = subprocess.run(command, cwd=ROOT, text=True, capture_output=True, timeout=240)
    assert process.returncode == 0, process.stderr
    summary = json.loads(process.stdout)
    assert summary["tilelang"] == {"TIMEOUT": 1}
    result_path = next((tmp_path / "timeout-run" / summary["run_id"]).glob("tilelang/*/*/result.json"))
    result = json.loads(result_path.read_text())
    assert result["phase"] == "pgir_parse"
    assert result["diagnostics"][-1]["code"] == "UPSTREAM_PGIR_PARSE_TIMEOUT"
    assert result["checker_loaded"] is False
