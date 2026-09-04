from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from pipeguard.checker import check_module
from pipeguard.frontends import triton_snapshot_to_module, tilelang_snapshot_to_module

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
TILELANG_PYTHON = Path(os.environ.get(
    "TILELANG_PYTHON",
    "/usr/local/miniconda3/envs/tl-pg/bin/python",
))
TRITON_PYTHON = Path(os.environ.get(
    "TRITON_PYTHON",
    "/usr/local/miniconda3/envs/triton-pg/bin/python",
))


def _run_capture(python: Path, tool: str, fixture: str, source: str, seed: int = 1):
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    return subprocess.run(
        [str(python), str(ROOT / "tools" / tool), str(FIXTURES / fixture), "--source", source],
        cwd=ROOT,
        env=environment,
        capture_output=True,
    )


def _capture(python: Path, tool: str, fixture: str, source: str, seed: int = 1) -> tuple[bytes, dict]:
    process = _run_capture(python, tool, fixture, source, seed)
    process.check_returncode()
    return process.stdout, json.loads(process.stdout)


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize(
    ("fixture", "source", "expected_prefix"),
    [
        ("tilelang_mbarrier_device.tvm.json", "live-tilelang-mbarrier", [
            "tilelang.mbarrier.init",
            "tilelang.barrier.arrive",
            "tilelang.barrier.observe",
        ]),
        ("tilelang_cp_async_device.tvm.json", "live-tilelang-cp-async", [
            "tilelang.mem.write",
            "tilelang.proxy_fence",
            "tilelang.cp_async.issue",
            "tilelang.cp_async.commit_group",
            "tilelang.cp_async.wait_group",
        ]),
    ],
)
def test_live_tilelang_capture_has_complete_v2_certificate_and_crosschecks(
    fixture: str,
    source: str,
    expected_prefix: list[str],
):
    _output, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py", fixture, source)
    assert snapshot["snapshot_version"] == "tilelang-snapshot-v2"
    assert [operation["op"] for operation in snapshot["operations"]][:len(expected_prefix)] == expected_prefix
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]
    assert snapshot["extraction"]["events_emitted"] == len(snapshot["operations"])
    assert all(operation["evidence"] for operation in snapshot["operations"])
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize(
    ("fixture", "source", "expected"),
    [
        ("triton_mbarrier.ttgir.mlir", "live-triton-mbarrier", [
            "triton.mbarrier.init",
            "triton.mbarrier.arrive",
            "triton.mbarrier.wait_parity",
        ]),
        ("triton_cp_async.ttgir.mlir", "live-triton-cp-async", [
            "triton.cp_async.issue",
            "triton.cp_async.commit_group",
            "triton.cp_async.wait_group",
            "triton.mem.read",
        ]),
    ],
)
def test_live_triton_capture_has_complete_v2_certificate_and_crosschecks(
    fixture: str,
    source: str,
    expected: list[str],
):
    _output, snapshot = _capture(TRITON_PYTHON, "triton_capture.py", fixture, source)
    assert snapshot["snapshot_version"] == "triton-snapshot-v2"
    assert [operation["op"] for operation in snapshot["operations"]] == expected
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]
    assert snapshot["extraction"]["events_emitted"] == len(snapshot["operations"])
    assert all(operation["evidence"] for operation in snapshot["operations"])
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_static_blocks_preserve_predicates_init_and_strided_offsets():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_static_block_device.tvm.json",
        "live-tilelang-static-block",
    )
    assert [operation.get("region") for operation in snapshot["operations"]] == [
        "region:global:source:52:4",
        "region:global:destination:52:4",
        "region:global:destination:4:4",
        "region:global:destination:12:4",
        "region:global:destination:44:4",
    ]
    assert all("statically_false_block" not in operation["evidence"]["source_location"]
               for operation in snapshot["operations"])
    assert sum("/init" in operation["evidence"]["source_location"]
               for operation in snapshot["operations"]) == 1
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_static_while_unfolds_bind_updates_to_exact_regions():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_static_while_device.tvm.json",
        "live-tilelang-static-while",
    )
    assert [operation["region"] for operation in snapshot["operations"]] == [
        "region:global:destination:4:4",
        "region:global:destination:36:4",
        "region:global:destination:68:4",
    ]
    assert snapshot["extraction"]["static_expansions"] == 3
    assert snapshot["extraction"]["elided_nodes"] == 4
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize(
    ("fixture", "code"),
    [
        ("tilelang_blackwell_device.tvm.json", "TILELANG_TARGET_UNSUPPORTED"),
        ("tilelang_amd_device.tvm.json", "TILELANG_TARGET_UNSUPPORTED"),
        ("tilelang_alias_unresolved_device.tvm.json", "TILELANG_ALIAS_UNRESOLVED"),
        ("tilelang_nested_effect_device.tvm.json", "TILELANG_EXPRESSION_EFFECT_UNSUPPORTED"),
    ],
)
def test_tilelang_capture_rejects_out_of_profile_targets_and_unknown_aliases(
    fixture: str,
    code: str,
):
    process = _run_capture(TILELANG_PYTHON, "tilelang_capture.py", fixture, "negative")
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["verdict"] == "UNSUPPORTED"
    assert result["diagnostics"][0]["code"] == code


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_large_static_tilelang_loop_uses_maximal_without_semantic_unroll_cap():
    first, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_static_loop_device.tvm.json",
        "live-tilelang-static-loop",
        seed=1,
    )
    second, _snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_static_loop_device.tvm.json",
        "live-tilelang-static-loop",
        seed=2,
    )
    assert first == second
    assert snapshot["extraction"]["static_expansions"] == 300
    assert snapshot["extraction"]["events_emitted"] == 600
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0
