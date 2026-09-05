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
    descriptor_evidence: str | None = None,
    binding_evidence: str | None = None,
):
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = FIXTURES / fixture_path
    command = [str(python), str(ROOT / "tools" / tool), str(fixture_path), "--source", source]
    if tool == "triton_capture.py":
        command.extend(["--grid-ctas", str(grid_ctas)])
    if descriptor_evidence is not None:
        command.extend(["--descriptor-evidence", str(FIXTURES / descriptor_evidence)])
    if binding_evidence is not None:
        command.extend(["--binding-evidence", str(FIXTURES / binding_evidence)])
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
    )


def _capture(
    python: Path,
    tool: str,
    fixture: str | Path,
    source: str,
    seed: int = 1,
    descriptor_evidence: str | None = None,
    binding_evidence: str | None = None,
) -> tuple[bytes, dict]:
    process = _run_capture(
        python,
        tool,
        fixture,
        source,
        seed,
        descriptor_evidence=descriptor_evidence,
        binding_evidence=binding_evidence,
    )
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
def test_triton_mbarrier_invalidate_reinitializes_existing_lifecycle_fsm():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_mbarrier_reuse.ttgir.mlir",
        "live-triton-mbarrier-reuse",
    )
    assert [(operation["op"], operation.get("lifecycle_epoch"))
            for operation in snapshot["operations"]] == [
        ("triton.mbarrier.init", 0),
        ("triton.mbarrier.arrive", 0),
        ("triton.mbarrier.wait_parity", 0),
        ("triton.mbarrier.invalidate", 0),
        ("triton.mbarrier.init", 1),
        ("triton.mbarrier.arrive", 1),
        ("triton.mbarrier.wait_parity", 1),
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_tma_load_maps_expect_issue_wait_to_existing_barrier_fsm():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_tma_load.ttgir.mlir",
        "live-triton-tma-load",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mbarrier.init",
        "triton.mbarrier.arrive_expect_tx",
        "triton.tma_load.issue",
        "triton.mbarrier.wait_parity",
        "triton.mem.read",
    ]
    assert snapshot["operations"][1]["bytes"] == 16384
    assert snapshot["operations"][2]["bytes"] == 16384
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_nonzero_tma_coordinates_use_exact_descriptor_segments():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_tma_load_nonzero.ttgir.mlir",
        "live-triton-tma-load-nonzero",
        binding_evidence="triton_tma_load_nonzero.binding.json",
    )
    issues = [
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.tma_load.issue"
    ]
    assert len(issues) == 64
    assert sum(issue["bytes"] for issue in issues) == 16384
    assert issues[0]["source_region"] == "region:global:global:tma:256:256"
    assert issues[-1]["source_region"] == "region:global:global:tma:32512:256"
    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["full_enabled_scans"] == 0


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_compiler_fenced_tma_store_maps_to_existing_typed_group_fsm():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_tma_store_fenced.ttgir.mlir",
        "live-triton-tma-store",
        binding_evidence="triton_tma_store_fenced.binding.json",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mem.write",
        "triton.proxy_fence",
        "triton.tma_store.issue",
        "triton.tma_store.commit_group",
        "triton.tma_store.wait_group",
    ]
    assert snapshot["operations"][-1]["wait_mode"] == "GLOBAL"
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_hidden_read_only_unit_attribute_requires_sha_bound_evidence():
    missing = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_tma_store_read_only_fenced.ttgir.mlir",
        "missing-binding-evidence",
    )
    assert missing.returncode == 2
    assert json.loads(missing.stdout)["diagnostics"][0]["code"] == "TRITON_BINDING_EVIDENCE_REQUIRED"

    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_tma_store_read_only_fenced.ttgir.mlir",
        "live-triton-tma-store-read-only",
        binding_evidence="triton_tma_store_read_only_fenced.binding.json",
    )
    assert snapshot["operations"][-1]["wait_mode"] == "READ_ONLY"
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_compiler_fenced_wgmma_maps_to_existing_typed_group_fsm():
    _output, fenced = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_wgmma_fenced.ttgir.mlir",
        "live-triton-wgmma-fenced",
    )
    assert [operation["op"] for operation in fenced["operations"]] == [
        "triton.mem.write",
        "triton.mem.write",
        "triton.proxy_fence",
        "triton.wgmma.fence_operand",
        "triton.wgmma.issue",
        "triton.wgmma.commit_group",
        "triton.wgmma.wait_group",
        "triton.wgmma.wait_group",
    ]
    waits = fenced["operations"][-2:]
    assert [wait["completion_kind"] for wait in waits] == [
        "OPERANDS_RELEASED",
        "RESULT_READY",
    ]
    assert check_module(triton_snapshot_to_module(fenced), engine="crosscheck")["verdict"] == "SAFE"

    _output, unfenced = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_wgmma.ttgir.mlir",
        "live-triton-wgmma-unfenced",
    )
    result = check_module(triton_snapshot_to_module(unfenced), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


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


def _scalar_pointer_ttgir(tmp_path: Path, arguments: str, body: str) -> Path:
    path = tmp_path / "scalar-pointer.ttgir.mlir"
    path.write_text(f"""\
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @scalar_pointer({arguments}) {{
{body}
    tt.return
  }}
}}
""")
    return path


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_pure_kernel_emits_empty_protocol_graph_and_checks_safe(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "",
        """    %one = arith.constant 1 : i32
    %two = arith.constant 2 : i32
    %sum = arith.addi %one, %two : i32""",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "pure-empty-protocol",
    )
    assert snapshot["operations"] == []
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]
    assert snapshot["extraction"]["classified_nodes"] > 0
    assert snapshot["extraction"]["emitting_nodes"] == 0
    assert snapshot["extraction"]["events_emitted"] == 0
    module = triton_snapshot_to_module(snapshot)
    assert check_module(module, engine="maximal")["verdict"] == "SAFE"
    assert check_module(module, engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_unknown_effect_is_not_misclassified_as_empty_protocol(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "",
        '    tt.print "unknown" { hex = false, isSigned = array<i32: 0>}',
    )
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "unknown-effect-not-empty",
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == "TRITON_OPERATION_UNSUPPORTED"


def _join_ttgir(tmp_path: Path, element_type: str, *, shared_input: bool = False) -> Path:
    path = tmp_path / "join.ttgir.mlir"
    arguments = f"%x: {element_type}" if shared_input else f"%x: {element_type}, %y: {element_type}"
    second_input = "%x" if shared_input else "%y"
    path.write_text(f"""\
#blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
#blocked1 = #ttg.blocked<{{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @value_join({arguments}) {{
    %a = tt.splat %x : {element_type} -> tensor<1x{element_type}, #blocked>
    %b = tt.splat {second_input} : {element_type} -> tensor<1x{element_type}, #blocked>
    %joined = tt.join %a, %b : tensor<1x{element_type}, #blocked> -> tensor<1x2x{element_type}, #blocked1>
    tt.return
  }}
}}
""")
    return path


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_value_join_is_protocol_pure(tmp_path: Path):
    fixture = _join_ttgir(tmp_path, "i32")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "value-join-pure",
    )
    assert snapshot["operations"] == []
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_value_split_and_unsplat_are_protocol_pure(tmp_path: Path):
    fixture = tmp_path / "split.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [1, 2], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @value_split() {
    %pair = arith.constant dense<[[1, 2]]> : tensor<1x2xi32, #blocked1>
    %left, %right = tt.split %pair : tensor<1x2xi32, #blocked1> -> tensor<1xi32, #blocked>
    %left_scalar = tt.unsplat %left : tensor<1xi32, #blocked>
    %right_scalar = tt.unsplat %right : tensor<1xi32, #blocked>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "value-split-unsplat-pure",
    )
    assert snapshot["operations"] == []
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_pointer_join_fails_closed(tmp_path: Path):
    fixture = _join_ttgir(tmp_path, "!tt.ptr<i32>", shared_input=True)
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "pointer-join-unsupported",
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == "TRITON_POINTER_PROVENANCE_UNRESOLVED"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_scalar_pointer_argument_tracks_direct_and_static_addptr_access(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%base: !tt.ptr<i32>",
        """    %zero = arith.constant 0 : i32
    tt.store %base, %zero : !tt.ptr<i32>
    %two = arith.constant 2 : i32
    %offset = tt.addptr %base, %two : !tt.ptr<i32>, i32
    %loaded = tt.load %offset : !tt.ptr<i32>""",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "scalar-pointer-direct-addptr",
    )
    assert [(operation["op"], operation["region"]) for operation in snapshot["operations"]] == [
        ("triton.mem.write", "region:global:global-arg:0:0:4"),
        ("triton.mem.read", "region:global:global-arg:0:8:4"),
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_unused_scalar_pointer_does_not_create_alias_obligation(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%used: !tt.ptr<i32>, %unused: !tt.ptr<i32>",
        """    %zero = arith.constant 0 : i32
    tt.store %used, %zero : !tt.ptr<i32>""",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "scalar-pointer-unused-argument",
    )
    assert [region["allocation_id"] for region in snapshot["regions"]] == ["global-arg:0"]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_multiple_used_scalar_pointers_still_require_alias_evidence(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%left: !tt.ptr<i32>, %right: !tt.ptr<i32>",
        """    %zero = arith.constant 0 : i32
    tt.store %left, %zero : !tt.ptr<i32>
    tt.store %right, %zero : !tt.ptr<i32>""",
    )
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "scalar-pointer-alias-required",
    )
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["diagnostics"][0]["code"] == "TRITON_ALIAS_UNRESOLVED"
    assert result["diagnostics"][0]["details"] == {"external_bases": 2}


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_used_subbyte_scalar_pointer_fails_closed(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%base: !tt.ptr<i1>",
        """    %false = arith.constant false
    tt.store %base, %false : !tt.ptr<i1>""",
    )
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "scalar-pointer-subbyte",
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == "TRITON_MEMORY_ACCESS_UNRESOLVED"


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
def test_triton_warp_specialize_uses_sha_bound_static_partition_evidence():
    missing = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_warp_specialize.ttgir.mlir",
        "missing-partition-evidence",
    )
    assert missing.returncode == 2
    assert json.loads(missing.stdout)["diagnostics"][0]["code"] == "TRITON_BINDING_EVIDENCE_REQUIRED"

    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_warp_specialize.ttgir.mlir",
        "live-triton-warp-specialize",
        binding_evidence="triton_warp_specialize.binding.json",
    )
    assert [(actor["id"], actor["kind"], len(actor["participants"]))
            for actor in snapshot["actors"]] == [
        ("cta", "WARPGROUP", 128),
        ("partition:0", "WARPGROUP", 128),
    ]
    assert [(operation["actor"], operation["op"])
            for operation in snapshot["operations"]] == [
        ("cta", "triton.mem.write"),
        ("cta", "triton.mem.read"),
        ("partition:0", "triton.mem.read"),
        ("cta", "triton.mem.read"),
    ]
    assert snapshot["operations"][2]["depends_on"] == [snapshot["operations"][0]["id"]]
    assert snapshot["operations"][3]["depends_on"] == [snapshot["operations"][2]["id"]]
    result = check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


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
def test_real_tilelang_tma_store_uses_elected_proxy_publication():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_tma_store_device.tvm.json",
        "live-tilelang-tma-store",
    )
    protocol = [
        operation["op"] for operation in snapshot["operations"]
        if operation["op"] in {
            "tilelang.proxy_fence",
            "tilelang.tma_store.issue",
            "tilelang.tma_store.commit_group",
            "tilelang.tma_store.wait_group",
        }
    ]
    assert protocol == [
        "tilelang.proxy_fence",
        "tilelang.tma_store.issue",
        "tilelang.tma_store.commit_group",
        "tilelang.tma_store.wait_group",
    ]
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["full_enabled_scans"] == 0
    assert result["statistics"]["state_clones"] == 0


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_tma_descriptor_evidence_is_bound_to_exact_ir(tmp_path: Path):
    evidence = json.loads(
        (FIXTURES / "tilelang_warp_specialized_descriptors.json").read_text()
    )
    evidence["source_ir_sha256"] = "0" * 64
    path = tmp_path / "descriptors.json"
    path.write_text(json.dumps(evidence))
    process = _run_capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_warp_specialized_device.tvm.json",
        "bad-descriptor-evidence",
        descriptor_evidence=str(path),
    )
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["diagnostics"][0]["code"] == "TILELANG_TMA_DESCRIPTOR_UNRESOLVED"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_real_tilelang_warp_specialized_tma_wgmma_capture_is_complete():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_warp_specialized_device.tvm.json",
        "live-tilelang-warp-specialized",
        descriptor_evidence="tilelang_warp_specialized_descriptors.json",
    )
    operation_types = {operation["op"] for operation in snapshot["operations"]}
    assert {
        "tilelang.mbarrier.expect_tx",
        "tilelang.tma_load.issue",
        "tilelang.wgmma.fence_operand",
        "tilelang.wgmma.issue",
        "tilelang.wgmma.commit_group",
        "tilelang.wgmma.wait_group",
    } <= operation_types
    assert {actor["kind"] for actor in snapshot["actors"]} == {"LANE", "WARPGROUP"}
    regions = {region["id"]: region for region in snapshot["regions"]}
    issue = next(operation for operation in snapshot["operations"]
                 if operation["op"] == "tilelang.wgmma.issue")
    accumulator = regions[issue["destination_region"]]
    assert any(
        operation["op"] == "tilelang.mem.read"
        and regions[operation["region"]]["allocation_id"] == accumulator["allocation_id"]
        for operation in snapshot["operations"]
    )
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]
    tilelang_snapshot_to_module(snapshot)


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
