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


def _run_capture(
    python: Path,
    tool: str,
    fixture: str | Path,
    source: str,
    seed: int = 1,
    grid_ctas: int = 1,
):
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = FIXTURES / fixture_path
    command = [str(python), str(ROOT / "tools" / tool), str(fixture_path), "--source", source]
    if tool == "triton_capture.py":
        command.extend(["--grid-ctas", str(grid_ctas)])
    return subprocess.run(
        command,
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
def test_triton_empty_commit_and_zero_token_wait_use_existing_group_fsm():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_empty_cp_group.ttgir.mlir",
        "live-triton-empty-cp-group",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.cp_async.commit_group",
        "triton.cp_async.wait_group",
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_masked_cp_async_splits_into_existing_issue_events():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_masked_cp_async.ttgir.mlir",
        "live-triton-masked-cp-async",
    )
    operations = [operation["op"] for operation in snapshot["operations"]]
    assert operations == ["triton.cp_async.issue"] * 8 + [
        "triton.cp_async.commit_group",
        "triton.cp_async.wait_group",
        "triton.mem.read",
    ]
    for offset, operation in enumerate(snapshot["operations"][:8]):
        assert operation["source_region"] == f"region:global:global-arg:0:{offset}:1"
        assert operation["destination_region"] == f"region:shared:shared:ssa:6:0:{offset}:1"
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_memdesc_index_preserves_allocation_and_static_subview_offset():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_memdesc_view.ttgir.mlir",
        "live-triton-memdesc-view",
    )
    assert [operation["region"] for operation in snapshot["operations"]] == [
        "region:shared:shared:ssa:1:0:16:16",
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_initialized_local_alloc_emits_shared_write_then_read():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_initialized_local.ttgir.mlir",
        "live-triton-initialized-local",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mem.write",
        "triton.mem.read",
    ]
    assert snapshot["operations"][0]["region"] == snapshot["operations"][1]["region"]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_ordinary_memory_ignores_dynamic_payload_but_keeps_effects():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_memory.ttgir.mlir",
        "live-triton-static-memory",
    )
    assert [(operation["op"], operation["region"]) for operation in snapshot["operations"]] == [
        ("triton.mem.read", "region:global:global-arg:0:0:16"),
        ("triton.mem.write", "region:global:global-arg:0:0:16"),
    ]
    assert snapshot["extraction"]["elided_nodes"] == 4
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize("target", ["cuda:100", "hip:gfx942"])
def test_triton_capture_rejects_non_sm90_targets(tmp_path: Path, target: str):
    module = (FIXTURES / "triton_static_memory.ttgir.mlir").read_text()
    path = tmp_path / "target.ttgir.mlir"
    path.write_text(module.replace('ttg.target = "cuda:90"', f'ttg.target = "{target}"'))
    process = _run_capture(TRITON_PYTHON, "triton_capture.py", path, "negative-target")
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["diagnostics"][0]["code"] == "TRITON_TARGET_UNSUPPORTED"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_capture_rejects_multi_cta_launch_evidence():
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_memory.ttgir.mlir",
        "multi-cta",
        grid_ctas=2,
    )
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["diagnostics"][0]["code"] == "TRITON_MULTI_CTA_UNSUPPORTED"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_dynamic_mask_fails_closed():
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_dynamic_mask.ttgir.mlir",
        "dynamic-mask",
    )
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["diagnostics"][0]["code"] == "TRITON_PREDICATE_UNRESOLVED"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_dynamic_control_fails_closed():
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_dynamic_control.ttgir.mlir",
        "dynamic-control",
    )
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["verdict"] == "UNSUPPORTED"
    assert result["diagnostics"][0]["code"] == "TRITON_SSA_VALUE_UNRESOLVED"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_mask_limits_memory_effect_region():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_masked_memory.ttgir.mlir",
        "live-triton-static-mask",
    )
    assert [(operation["op"], operation["region"]) for operation in snapshot["operations"]] == [
        ("triton.mem.read", "region:global:global-arg:0:0:8"),
        ("triton.mem.write", "region:global:global-arg:0:0:8"),
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_disjoint_static_footprint_splits_exact_regions():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_disjoint_memory.ttgir.mlir",
        "live-triton-static-disjoint-memory",
    )
    assert [operation["region"] for operation in snapshot["operations"]] == [
        f"region:global:global-arg:0:{offset}:1" for offset in range(0, 32, 2)
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_loop_carried_pointer_offsets_resolve_per_activation():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_pointer_loop.ttgir.mlir",
        "live-triton-static-pointer-loop",
    )
    assert [operation["region"] for operation in snapshot["operations"]] == [
        "region:global:global-arg:0:0:1",
        "region:global:global-arg:0:1:1",
        "region:global:global-arg:0:2:1",
    ]
    assert snapshot["extraction"]["static_expansions"] == 3
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_while_unfolds_carried_ssa_values():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_while.ttgir.mlir",
        "live-triton-static-while",
    )
    assert len(snapshot["operations"]) == 3
    assert all(operation["op"] == "triton.mem.read" for operation in snapshot["operations"])
    assert snapshot["extraction"]["static_expansions"] == 3
    assert [operation["evidence"]["activation_path"][3]
            for operation in snapshot["operations"]] == [0, 1, 2]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_regions_unfold_for_and_select_if_paths():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_control.ttgir.mlir",
        "live-triton-static-control",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mem.read",
        "triton.mem.read",
        "triton.mem.read",
        "triton.mem.read",
        "triton.mem.read",
    ]
    assert snapshot["extraction"]["static_expansions"] == 3
    paths = [operation["evidence"]["activation_path"] for operation in snapshot["operations"]]
    assert [path[1] for path in paths[:3]] == ["for", "for", "for"]
    assert paths[3][1] == "if" and paths[3][3] == 1
    assert paths[4][1] == "if" and paths[4][3] == 0
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
