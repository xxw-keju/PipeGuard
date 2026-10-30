from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from pipeguard.checker import check_module
from pipeguard.frontends import FrontendError, triton_snapshot_to_module, tilelang_snapshot_to_module

ROOT = Path(__file__).parents[1]
FIXTURES = ROOT / "tests" / "fixtures"
sys.path.insert(0, str(ROOT / "tools"))
from tilelang_capture import (  # noqa: E402
    CaptureError as TileLangCaptureError,
    _prove_wgmma_group,
    _wgmma_issue_operand,
    _wgmma_layout_hull,
)

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
    grid_ctas: int | None = 1,
    launch_grid: tuple[int, ...] | None = None,
    selected_cta: tuple[int, int, int] = (0, 0, 0),
    descriptor_evidence: str | None = None,
    binding_evidence: str | Path | None = None,
    final_ptx: str | Path | None = None,
):
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = FIXTURES / fixture_path
    command = [str(python), str(ROOT / "tools" / tool), str(fixture_path), "--source", source]
    if tool == "tilelang_capture.py" and launch_grid is not None:
        command.extend(["--launch-grid", *(str(extent) for extent in launch_grid)])
        command.extend(["--selected-cta", *(str(index) for index in selected_cta)])
    if tool == "triton_capture.py":
        if grid_ctas is not None:
            command.extend(["--grid-ctas", str(grid_ctas)])
        if launch_grid is not None:
            command.extend(["--launch-grid", *(str(extent) for extent in launch_grid)])
        command.extend(["--selected-cta", *(str(index) for index in selected_cta)])
    if descriptor_evidence is not None:
        command.extend(["--descriptor-evidence", str(FIXTURES / descriptor_evidence)])
    if binding_evidence is not None:
        evidence_path = Path(binding_evidence)
        if not evidence_path.is_absolute():
            evidence_path = FIXTURES / evidence_path
        command.extend(["--binding-evidence", str(evidence_path)])
    if final_ptx is not None:
        final_ptx_path = Path(final_ptx)
        if not final_ptx_path.is_absolute():
            final_ptx_path = FIXTURES / final_ptx_path
        command.extend(["--final-ptx", str(final_ptx_path)])
    return subprocess.run(
        command,
        cwd=ROOT,
        env=environment,
        capture_output=True,
    )


def _compact_field_values(operation: dict, field: str) -> list:
    if field in operation.get("fields", {}):
        return [operation["fields"][field]] * len(operation["instances"])
    return list(operation.get("instance_fields", {}).get(field, []))


def _capture(
    python: Path,
    tool: str,
    fixture: str | Path,
    source: str,
    seed: int = 1,
    grid_ctas: int | None = 1,
    launch_grid: tuple[int, ...] | None = None,
    selected_cta: tuple[int, int, int] = (0, 0, 0),
    descriptor_evidence: str | None = None,
    binding_evidence: str | Path | None = None,
    final_ptx: str | Path | None = None,
) -> tuple[bytes, dict]:
    process = _run_capture(
        python,
        tool,
        fixture,
        source,
        seed,
        grid_ctas=grid_ctas,
        launch_grid=launch_grid,
        selected_cta=selected_cta,
        descriptor_evidence=descriptor_evidence,
        binding_evidence=binding_evidence,
        final_ptx=final_ptx,
    )
    process.check_returncode()
    return process.stdout, json.loads(process.stdout)


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_predicated_bar_sync_inline_asm_is_captured_exactly():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_effectful_inline_asm.ttgir.mlir",
        "triton-predicated-bar-sync-inline-asm",
    )

    barriers = [
        operation for operation in snapshot["operations"]
        if operation["op"].startswith("triton.barrier.")
    ]
    assert [operation["op"] for operation in barriers] == [
        "triton.barrier.arrive",
        "triton.barrier.observe",
    ]
    assert barriers[0]["count"] == 1
    assert barriers[0]["execution_evidence"]["program_instances"] == list(range(16))
    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "STUCK_PROTOCOL_STATE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_unknown_effectful_inline_asm_still_fails_closed(tmp_path: Path):
    fixture = tmp_path / "unknown-effectful-inline-asm.ttgir.mlir"
    fixture.write_text(
        (FIXTURES / "triton_effectful_inline_asm.ttgir.mlir").read_text().replace(
            "@p bar.sync 0;",
            "@p bar.warp.sync 0xffffffff;",
        )
    )
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "triton-unknown-effectful-inline-asm",
    )

    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == (
        "TRITON_EFFECTFUL_INLINE_ASM_UNSUPPORTED"
    )


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_parameterized_profile_requires_lowered_execution_evidence():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_mbarrier.ttgir.mlir",
        "triton-parameterized-profile-contract",
    )
    assert snapshot["profile"]["id"] == "sm90-single-cta-static-parametric-v1"
    module = triton_snapshot_to_module(snapshot)
    assert module.raw["schema_version"] == "pgir-parametric-v1"
    assert check_module(module, engine="maximal")["statistics"]["representation"] == "parametric-protocol"

    incomplete = json.loads(json.dumps(snapshot))
    incomplete["operations"][0].pop("execution_evidence")
    with pytest.raises(FrontendError) as caught:
        triton_snapshot_to_module(incomplete)
    assert caught.value.code == "TRITON_EXECUTION_SCOPE_UNRESOLVED"


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
def test_live_tilelang_capture_has_complete_v3_certificate_with_maximal(
    fixture: str,
    source: str,
    expected_prefix: list[str],
):
    _output, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py", fixture, source)
    assert snapshot["snapshot_version"] == "tilelang-snapshot-v3"
    assert [operation["op"] for operation in snapshot["operations"]][:len(expected_prefix)] == expected_prefix
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]
    assert snapshot["extraction"]["operation_records"] == len(snapshot["operations"])
    assert snapshot["extraction"]["logical_events_emitted"] == sum(
        len(operation["instances"]) for operation in snapshot["operations"]
    )
    assert all(operation["evidence"] for operation in snapshot["operations"])
    if fixture == "tilelang_mbarrier_device.tvm.json":
        assert {tuple(operation["instances"]) for operation in snapshot["operations"]} >= {
            (0,),
            (1,),
            (0, 1),
        }
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["representation"].startswith("parametric")


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
        ("triton.barrier.arrive", None),
        ("triton.barrier.observe", None),
        ("triton.mbarrier.arrive", 0),
        ("triton.mbarrier.wait_parity", 0),
        ("triton.mbarrier.invalidate", 0),
        ("triton.mbarrier.init", 1),
        ("triton.barrier.arrive", None),
        ("triton.barrier.observe", None),
        ("triton.mbarrier.arrive", 1),
        ("triton.mbarrier.wait_parity", 1),
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_initial_opposite_parity_wait_is_a_control_only_priming_poll(tmp_path: Path):
    module = (FIXTURES / "triton_mbarrier.ttgir.mlir").read_text()
    module = module.replace("%phase = arith.constant 0 : i32", "%phase = arith.constant 1 : i32")
    module = module.replace(
        "    ttng.arrive_barrier %bar, 1, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>\n",
        "",
    )
    fixture = tmp_path / "primed-mbarrier.ttgir.mlir"
    fixture.write_text(module)
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "primed-mbarrier",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mbarrier.init",
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_unconditional_arrive_barrier_defaults_to_true(tmp_path: Path):
    module = (FIXTURES / "triton_mbarrier.ttgir.mlir").read_text().replace(
        "ttng.arrive_barrier %bar, 1, %true",
        "ttng.arrive_barrier %bar, 1",
    )
    fixture = tmp_path / "unconditional-arrive.ttgir.mlir"
    fixture.write_text(module)
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "unconditional-arrive",
    )
    arrival = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.mbarrier.arrive"
    )
    assert arrival["predicate"] is True
    assert arrival["count"] == 1
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_memdesc_indices_identify_distinct_barriers(tmp_path: Path):
    fixture = tmp_path / "indexed-mbarriers.ttgir.mlir"
    fixture.write_text("""\
#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @indexed_mbarriers() {
    %bars = ttg.local_alloc : () -> !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    %zero = arith.constant 0 : i32
    %one = arith.constant 1 : i32
    %true = arith.constant true
    %phase = arith.constant 0 : i32
    %bar0 = ttg.memdesc_index %bars[%zero] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %bar1 = ttg.memdesc_index %bars[%one] : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable> -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar0, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.arrive_barrier %bar0, 1, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.wait_barrier %bar0, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.init_barrier %bar1, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.arrive_barrier %bar1, 1, %true : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.wait_barrier %bar1, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttg.local_dealloc %bars : !ttg.memdesc<2x1xi64, #barrier, #smem, mutable>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "indexed-mbarriers",
    )
    resources = [operation["resource"] for operation in snapshot["operations"]
                 if operation["op"].startswith("triton.mbarrier.")]
    assert len(set(resources[:3])) == 1
    assert len(set(resources[3:])) == 1
    assert resources[0] != resources[3]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_local_collective_barrier_uses_dsl_neutral_cta_protocol(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "",
        "    ttg.barrier local",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "local-collective-barrier",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.barrier.arrive",
        "triton.barrier.observe",
    ]
    resource = snapshot["resources"][0]
    assert resource["kind"] == "cta_barrier"
    assert resource["capacity"] == 128
    assert resource["visibility_mask"]["shared"] is True
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"

    partial = tmp_path / "partial-collective.ttgir.mlir"
    partial.write_text(fixture.read_text().replace(
        '"ttg.num-warps" = 4 : i32,',
        '"ttg.num-warps" = 4 : i32, "ttg.total-num-warps" = 8 : i32,',
    ))
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        partial,
        "partial-collective-barrier",
    )
    assert snapshot["target"]["cta_threads"] == 256
    assert snapshot["resources"][0]["kind"] == "named_barrier"
    assert snapshot["resources"][0]["capacity"] == 128
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_tma_load_maps_expect_issue_wait_to_existing_barrier_fsm(tmp_path: Path):
    fixture = FIXTURES / "triton_tma_load.ttgir.mlir"
    missing = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "missing-tma-layout-evidence",
    )
    assert missing.returncode == 2
    diagnostic = json.loads(missing.stdout)["diagnostics"][0]
    assert diagnostic["code"] == "TRITON_TMA_DESCRIPTOR_UNRESOLVED"
    operation_index = diagnostic["details"]["operation_index"]
    evidence = tmp_path / "tma-layout.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {str(operation_index): {"descriptor": {
            "allocation_id": "global:tma",
            "element_bytes": 4,
            "shape": [64, 64],
            "strides": [64, 1],
            "tile_shape": [64, 64],
            "coordinate_order": [1, 0],
        }}},
    }))
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "live-triton-tma-load",
        binding_evidence=evidence,
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
    assert snapshot["operations"][2]["transfers"]
    assert "source_region" not in snapshot["operations"][2]
    assert "destination_region" not in snapshot["operations"][2]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


def _capture_with_tma_layout(tmp_path: Path, fixture: Path, source: str, allocation: str):
    # Supplies the descriptor layout evidence for every TMA operation the
    # capture reports as unresolved.
    attributes: dict[str, dict] = {}
    evidence = tmp_path / f"{source}.evidence.json"
    while True:
        evidence.write_text(json.dumps({
            "evidence_version": "triton-binding-evidence-v1",
            "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
            "unit_attributes": {},
            "array_attributes": attributes,
        }))
        process = _run_capture(
            TRITON_PYTHON, "triton_capture.py", fixture, source, binding_evidence=evidence
        )
        if process.returncode == 0:
            return _capture(
                TRITON_PYTHON, "triton_capture.py", fixture, source, binding_evidence=evidence
            )
        diagnostic = json.loads(process.stdout)["diagnostics"][0]
        assert diagnostic["code"] == "TRITON_TMA_DESCRIPTOR_UNRESOLVED", diagnostic
        key = str(diagnostic["details"]["operation_index"])
        assert key not in attributes
        attributes[key] = {"descriptor": {
            "allocation_id": allocation,
            "element_bytes": 4,
            "shape": [64, 64],
            "strides": [64, 1],
            "tile_shape": [64, 64],
            "coordinate_order": [1, 0],
        }}


_TMA_BARRIER_TYPE = "!ttg.memdesc<1xi64, #barrier, #smem, mutable>"
_TMA_LOAD_LINE = (
    "    ttng.async_tma_copy_global_to_local %descriptor[%zero, %zero] %destination, %barrier_arg, %true"
    " : !tt.tensordesc<64x64xf32, #shared>, " + _TMA_BARRIER_TYPE
    + " -> !ttg.memdesc<64x64xf32, #shared, #smem, mutable>\n"
)


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_tma_load_of_a_signed_integer_descriptor(tmp_path: Path):
    # Tensor descriptors spell signed integers si32; memdescs use signless i32.
    text = (FIXTURES / "triton_tma_load.ttgir.mlir").read_text()
    text = text.replace("xf32", "xi32").replace("tensordesc<64x64xi32", "tensordesc<64x64xsi32")
    fixture = tmp_path / "tma-load-si32.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture_with_tma_layout(tmp_path, fixture, "tma-load-si32", "global:tma")
    ops = [operation["op"] for operation in snapshot["operations"]]
    assert "triton.tma_load.issue" in ops
    assert snapshot["operations"][ops.index("triton.mbarrier.arrive_expect_tx")]["bytes"] == 16384
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_tma_load_reuse_gets_war_membar(tmp_path: Path):
    # Reloading the tile every thread has just read is ordered only by the
    # barrier Membar inserts before the TMA write (WAR on the shared tile).
    text = (FIXTURES / "triton_tma_load.ttgir.mlir").read_text()
    reload = (
        f"    ttng.barrier_expect %barrier_arg, 16384, %true : {_TMA_BARRIER_TYPE}\n"
        + _TMA_LOAD_LINE
        + "    %one = arith.constant 1 : i32\n"
        + f"    ttng.wait_barrier %barrier_arg, %one : {_TMA_BARRIER_TYPE}\n"
        + "    %reloaded = ttg.local_load %destination : !ttg.memdesc<64x64xf32, #shared, #smem, mutable>"
        + " -> tensor<64x64xf32, #blocked>\n"
    )
    text = text.replace("    tt.return\n", reload + "    tt.return\n")
    fixture = tmp_path / "tma-load-reuse.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture_with_tma_layout(tmp_path, fixture, "tma-load-reuse", "global:tma")
    ops = [operation["op"] for operation in snapshot["operations"]]
    first_read = ops.index("triton.mem.read")
    second_issue = ops.index("triton.tma_load.issue", first_read)
    assert "triton.barrier.arrive" in ops[first_read:second_issue]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize(("source", "verdict"), [("global:tma", "SAFE"), ("global-arg:2", "DATA_HAZARD")])
def test_triton_fence_async_shared_does_not_publish_global_writes_to_tma(
    tmp_path: Path,
    source: str,
    verdict: str,
):
    # fence.proxy.async.shared::cta orders shared state only, so a generic
    # store into the TMA source is not made visible to the async proxy.
    text = (FIXTURES / "triton_tma_load.ttgir.mlir").read_text()
    text = text.replace(
        "#smem = #ttg.shared_memory",
        "#smem = #ttg.shared_memory\n"
        "#linear = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>",
    ).replace(
        f"%barrier_arg: {_TMA_BARRIER_TYPE})",
        f"%barrier_arg: {_TMA_BARRIER_TYPE}, %out: !tt.ptr<f32>)",
    ).replace(
        _TMA_LOAD_LINE,
        "    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #linear>\n"
        "    %out_vector = tt.splat %out : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #linear>\n"
        "    %addresses = tt.addptr %out_vector, %range : tensor<128x!tt.ptr<f32>, #linear>, tensor<128xi32, #linear>\n"
        "    %ones = arith.constant dense<1.0> : tensor<128xf32, #linear>\n"
        "    tt.store %addresses, %ones : tensor<128x!tt.ptr<f32>, #linear>\n"
        "    ttg.barrier local\n"
        "    ttng.fence_async_shared {bCluster = false}\n"
        + _TMA_LOAD_LINE,
    )
    fixture = tmp_path / "tma-load-after-store.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture_with_tma_layout(tmp_path, fixture, "tma-load-after-store", source)
    fence = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.proxy_fence"
    )
    assert fence["visibility_mask"]["shared"] is True
    assert fence["visibility_mask"]["global_write"] is False
    assert all(region.startswith("region:shared:") for region in fence["regions"])
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == verdict


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
    assert len(issues) == 1
    assert issues[0]["bytes"] == 16384
    assert len(issues[0]["transfers"]) == 64
    assert sum(transfer["bytes"] for transfer in issues[0]["transfers"]) == 16384
    assert issues[0]["transfers"][0]["source_region"] == "region:global:global:tma:256:256"
    assert issues[0]["transfers"][-1]["source_region"] == "region:global:global:tma:32512:256"
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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_hidden_read_only_unit_attribute_requires_sha_bound_evidence():
    missing = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_tma_store_read_only_fenced.ttgir.mlir",
        "missing-binding-evidence",
    )
    assert missing.returncode == 2
    assert json.loads(missing.stdout)["diagnostics"][0]["code"] == "TRITON_TMA_DESCRIPTOR_UNRESOLVED"

    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_tma_store_read_only_fenced.ttgir.mlir",
        "live-triton-tma-store-read-only",
        binding_evidence="triton_tma_store_read_only_fenced.binding.json",
    )
    assert snapshot["operations"][-1]["wait_mode"] == "READ_ONLY"
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(fenced), engine="maximal")["verdict"] == "SAFE"

    _output, unfenced = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_wgmma.ttgir.mlir",
        "live-triton-wgmma-unfenced",
    )
    result = check_module(triton_snapshot_to_module(unfenced), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize(
    ("fence_before", "first_op"),
    [
        ("%0 = ttg.local_alloc", "triton.proxy_fence"),
        ("%1 = ttg.local_alloc", "triton.mem.write"),
    ],
)
def test_triton_fence_before_generic_write_does_not_publish_it(
    tmp_path: Path,
    fence_before: str,
    first_op: str,
):
    fence = "    ttng.fence_async_shared {bCluster = false}\n"
    text = (FIXTURES / "triton_wgmma_fenced.ttgir.mlir").read_text().replace(fence, "")
    text = text.replace(f"    {fence_before}", fence + f"    {fence_before}")
    fixture = tmp_path / "wgmma-misplaced-fence.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "live-triton-wgmma-misplaced-fence",
    )
    operations = [operation["op"] for operation in snapshot["operations"]]
    assert operations[0] == first_op
    assert operations.index("triton.proxy_fence") < operations.index("triton.wgmma.issue")
    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_final_ptx_proxy_fence_is_sha_bound_and_restores_wgmma_order(
    tmp_path: Path,
):
    final_ptx = tmp_path / "kernel.ptx"
    final_ptx.write_text(""".visible .entry kernel() {
    fence.proxy.async.shared::cta;
    wgmma.fence.sync.aligned;
    wgmma.mma_async.sync.aligned.m64n8k16.f32.f16.f16 {%r0,%r1,%r2,%r3}, %rd0, %rd1, 0, 1, 1, 0, 0;
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_wgmma.ttgir.mlir",
        "live-triton-wgmma-final-ptx-fenced",
        final_ptx=final_ptx,
    )

    assert [operation["op"] for operation in snapshot["operations"]].count(
        "triton.proxy_fence"
    ) == 1
    evidence = snapshot["profile"]["final_ptx_evidence"]
    assert evidence["sha256"] == hashlib.sha256(final_ptx.read_bytes()).hexdigest()
    assert evidence["proxy_fence_before_first_wgmma"] is True
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")[
        "verdict"
    ] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_final_ptx_proxy_fence_rejects_ambiguous_mapping(tmp_path: Path):
    final_ptx = tmp_path / "ambiguous.ptx"
    final_ptx.write_text(""".visible .entry kernel() {
    fence.proxy.async.shared::cta;
    fence.proxy.async.shared::cta;
    wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16;
}
""")
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_wgmma.ttgir.mlir",
        "live-triton-wgmma-final-ptx-ambiguous",
        final_ptx=final_ptx,
    )

    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == (
        "TRITON_FINAL_PTX_PROXY_FENCE_AMBIGUOUS"
    )


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_empty_wgmma_wait_is_protocol_pure(tmp_path: Path):
    fixture = tmp_path / "empty-wgmma-wait.ttgir.mlir"
    fixture.write_text("""\
#mma = #ttg.nvidia_mma<{versionMajor = 3, versionMinor = 0, warpsPerCTA = [4, 1], instrShape = [16, 64, 16]}>
module attributes {"ttg.target" = "cuda:90", "ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @empty_wgmma_wait() {
    %acc = arith.constant dense<0.000000e+00> : tensor<128x64xf32, #mma>
    %waited = ttng.warp_group_dot_wait %acc {pendings = 0 : i32} : tensor<128x64xf32, #mma>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "empty-wgmma-wait",
    )
    assert snapshot["operations"] == []


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_wgmma_accepts_uniform_multi_warpgroup_partition(tmp_path: Path):
    module = (FIXTURES / "triton_wgmma_fenced.ttgir.mlir").read_text()
    module = module.replace("warpsPerCTA = [4, 1]", "warpsPerCTA = [8, 1]")
    module = module.replace("warpsPerCTA = [2, 2]", "warpsPerCTA = [4, 2]")
    module = module.replace("128x64", "256x64")
    module = module.replace('"ttg.num-warps" = 4 : i32', '"ttg.num-warps" = 8 : i32')
    fixture = tmp_path / "multi-warpgroup-wgmma.ttgir.mlir"
    fixture.write_text(module)
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "multi-warpgroup-wgmma",
    )
    assert snapshot["actors"][0]["kind"] == "PARTITION"
    assert len(snapshot["actors"][0]["participants"]) == 256
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_loop_carried_wgmma_accumulator_keeps_register_identity(tmp_path: Path):
    module = (FIXTURES / "triton_wgmma_fenced.ttgir.mlir").read_text()
    module = module.replace(
        """    %2 = ttng.warp_group_dot %0, %1, %cst_1 {isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
    %3 = ttng.warp_group_dot_wait %2 {pendings = 0 : i32} : tensor<128x64xf32, #mma>""",
        """    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %3 = scf.for %i = %lower to %upper step %step iter_args(%acc = %cst_1) -> (tensor<128x64xf32, #mma>) {
      %2 = ttng.warp_group_dot %0, %1, %acc {isAsync = true} : !ttg.memdesc<128x64xf16, #shared, #smem> * !ttg.memdesc<64x64xf16, #shared1, #smem> -> tensor<128x64xf32, #mma>
      %waited = ttng.warp_group_dot_wait %2 {pendings = 1 : i32} : tensor<128x64xf32, #mma>
      scf.yield %waited : tensor<128x64xf32, #mma>
    }
    %done = ttng.warp_group_dot_wait %3 {pendings = 0 : i32} : tensor<128x64xf32, #mma>""",
    )
    fixture = tmp_path / "loop-carried-wgmma.ttgir.mlir"
    fixture.write_text(module)
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "loop-carried-wgmma",
    )
    issues = [operation for operation in snapshot["operations"] if operation["op"] == "triton.wgmma.issue"]
    assert len(issues) == 2
    assert [issue["engine_seq"] for issue in issues] == [0, 1]
    assert len({issue["engine_id"] for issue in issues}) == 1
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_wgmma_result_use_requires_result_ready_wait(tmp_path: Path):
    module = (FIXTURES / "triton_wgmma_fenced.ttgir.mlir").read_text()
    waited = module.replace(
        "    tt.return",
        "    %used = arith.addf %3, %3 : tensor<128x64xf32, #mma>\n    tt.return",
    )
    waited_fixture = tmp_path / "wgmma-result-waited.ttgir.mlir"
    waited_fixture.write_text(waited)
    _output, waited_snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        waited_fixture,
        "wgmma-result-waited",
    )
    assert waited_snapshot["operations"][-1]["op"] == "triton.mem.read"
    assert check_module(triton_snapshot_to_module(waited_snapshot), engine="maximal")["verdict"] == "SAFE"

    missing = module.replace(
        "    %3 = ttng.warp_group_dot_wait %2 {pendings = 0 : i32} : tensor<128x64xf32, #mma>\n",
        "",
    ).replace(
        "    tt.return",
        "    %used = arith.addf %2, %2 : tensor<128x64xf32, #mma>\n    tt.return",
    )
    missing_fixture = tmp_path / "wgmma-result-missing-wait.ttgir.mlir"
    missing_fixture.write_text(missing)
    _output, missing_snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        missing_fixture,
        "wgmma-result-missing-wait",
    )
    result = check_module(triton_snapshot_to_module(missing_snapshot), engine="maximal")
    assert missing_snapshot["operations"][-1]["region"].startswith("region:register:")
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "WGMMA_RESULT_NOT_READY"
    assert [finding["class"] for finding in result["all_findings"]] == [
        "SYNC_HAZARD",
        "DATA_HAZARD",
    ]
    assert result["all_findings"][1]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_synchronous_wgmma_waits_for_its_result(tmp_path: Path):
    module = (FIXTURES / "triton_wgmma_fenced.ttgir.mlir").read_text()
    module = module.replace(
        "%cst_1 {isAsync = true} :",
        "%cst_1 :",
    ).replace(
        "    %3 = ttng.warp_group_dot_wait %2 {pendings = 0 : i32} : tensor<128x64xf32, #mma>\n",
        "",
    ).replace(
        "    tt.return",
        "    %used = arith.addf %2, %2 : tensor<128x64xf32, #mma>\n    tt.return",
    )
    assert "isAsync" not in module
    fixture = tmp_path / "wgmma-synchronous.ttgir.mlir"
    fixture.write_text(module)
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "wgmma-synchronous",
    )
    ops = [operation["op"] for operation in snapshot["operations"]]
    commit = ops.index("triton.wgmma.commit_group")
    assert ops[commit + 1 : commit + 3] == ["triton.wgmma.wait_group", "triton.wgmma.wait_group"]
    assert [
        operation["completion_kind"]
        for operation in snapshot["operations"][commit + 1 : commit + 3]
    ] == ["OPERANDS_RELEASED", "RESULT_READY"]
    assert snapshot["operations"][-1]["op"] == "triton.mem.read"
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_sha_bound_final_lowering_wait_orders_wgmma_before_arrive(tmp_path: Path):
    module = (FIXTURES / "triton_wgmma_fenced.ttgir.mlir").read_text()
    module = module.replace(
        "#smem = #ttg.shared_memory",
        "#barrier = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>\n#smem = #ttg.shared_memory",
    ).replace(
        "  tt.func public @wgmma_roundtrip() {",
        """  tt.func public @wgmma_roundtrip() {
    %bar = ttg.local_alloc : () -> !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    %phase = arith.constant 0 : i32
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>""",
    ).replace(
        "    %3 = ttng.warp_group_dot_wait %2 {pendings = 0 : i32} : tensor<128x64xf32, #mma>",
        """    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>
    ttng.wait_barrier %bar, %phase : !ttg.memdesc<1xi64, #barrier, #smem, mutable>""",
    )
    fixture = tmp_path / "implicit-wgmma-wait.ttgir.mlir"
    fixture.write_text(module)
    missing = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "implicit-wgmma-wait-missing",
    )
    assert missing.returncode == 2
    diagnostic = json.loads(missing.stdout)["diagnostics"][0]
    assert diagnostic["code"] == "TRITON_BINDING_EVIDENCE_REQUIRED"
    operation_index = diagnostic["details"]["operation_index"]

    evidence = tmp_path / "implicit-wgmma-wait.binding.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {str(operation_index): ["implicit_wgmma_wait_group_0"]},
        "array_attributes": {},
    }))
    process = subprocess.run(
        [
            str(TRITON_PYTHON),
            str(ROOT / "tools" / "triton_capture.py"),
            str(fixture),
            "--source",
            "implicit-wgmma-wait",
            "--grid-ctas",
            "1",
            "--selected-cta",
            "0",
            "0",
            "0",
            "--binding-evidence",
            str(evidence),
        ],
        cwd=ROOT,
        capture_output=True,
    )
    process.check_returncode()
    snapshot = json.loads(process.stdout)
    arrival = next(
        index for index, operation in enumerate(snapshot["operations"])
        if operation["op"] == "triton.mbarrier.arrive"
    )
    waits = snapshot["operations"][arrival - 2:arrival]
    assert [wait["op"] for wait in waits] == [
        "triton.wgmma.wait_group",
        "triton.wgmma.wait_group",
    ]
    assert {wait["completion_kind"] for wait in waits} == {
        "OPERANDS_RELEASED",
        "RESULT_READY",
    }
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_expands_static_void_private_function_call(tmp_path: Path):
    fixture = tmp_path / "static-private-call.ttgir.mlir"
    fixture.write_text("""module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @entry(%ptr: !tt.ptr<i32>) {
    tt.call @helper(%ptr) : (!tt.ptr<i32>) -> ()
    tt.return
  }
  tt.func private @helper(%ptr: !tt.ptr<i32>) {
    %value = tt.load %ptr : !tt.ptr<i32>
    %group = ttg.async_commit_group
    %wait = ttg.async_wait {num = 0 : i32}
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "static-private-call",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mem.read",
        "triton.cp_async.commit_group",
        "triton.cp_async.wait_group",
        "triton.barrier.arrive",
        "triton.barrier.observe",
    ]
    assert snapshot["extraction"]["static_expansions"] == 1
    module = triton_snapshot_to_module(snapshot)
    assert check_module(module, engine="maximal")["verdict"] == "SAFE"


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
        "triton.barrier.arrive",
        "triton.barrier.observe",
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_cp_async_wait_12_preserves_threshold_and_leaves_one_group_pending(tmp_path: Path):
    module = (FIXTURES / "triton_cp_async.ttgir.mlir").read_text()
    fixture = tmp_path / "cp-async-wait-12.ttgir.mlir"
    fixture.write_text(module.replace("{num = 0 : i32}", "{num = 12 : i32}"))
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "cp-async-wait-12",
    )
    wait = next(operation for operation in snapshot["operations"] if operation["op"] == "triton.cp_async.wait_group")
    assert wait["wait_group_count"] == 12
    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_masked_cp_async_coalesces_contiguous_issue_segments():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_masked_cp_async.ttgir.mlir",
        "live-triton-masked-cp-async",
    )
    operations = [operation["op"] for operation in snapshot["operations"]]
    assert operations == [
        "triton.cp_async.issue",
        "triton.cp_async.commit_group",
        "triton.cp_async.wait_group",
        "triton.barrier.arrive",
        "triton.barrier.observe",
        "triton.mem.read",
    ]
    issue = snapshot["operations"][0]
    # Masked-off elements are issued with src-size 0: no global read, but
    # the shared destination is still zero-filled.
    assert issue["source_region"] == "region:global:global-arg:0:0:32"
    assert issue["destination_region"] == "region:shared:shared:ssa:6:0:0:64"
    assert issue["bytes"] == 64
    assert issue["lane_effects"]["lanes"] == list(range(16))
    assert issue["lane_effects"]["bytes"] == [4] * 16
    assert issue["lane_effects"]["source_regions"][8:] == [
        "region:global:cp-async-zfill-no-read:0:4"
    ] * 8
    assert issue["execution_evidence"]["transition"] == "PER_INSTANCE"
    assert issue["execution_evidence"]["rendezvous"] == "NONE"
    module = triton_snapshot_to_module(snapshot)
    cp_templates = [
        template for template in module.templates if template.type.startswith("CpAsync")
    ]
    assert cp_templates[0].program_instances == tuple(range(16))
    assert cp_templates[0].transition == "PER_INSTANCE"
    assert cp_templates[0].rendezvous == "NONE"
    assert cp_templates[1].program_instances == tuple(range(128))
    assert cp_templates[2].program_instances == tuple(range(128))
    assert check_module(module, engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize(("waited", "verdict"), [(True, "SAFE"), (False, "DATA_HAZARD")])
def test_triton_fully_masked_cp_async_still_zero_fills_shared(
    tmp_path: Path,
    waited: bool,
    verdict: str,
):
    # A false mask (e.g. a pipeliner stage predicate) lowers to cp.async with
    # src-size 0, which still writes zeros asynchronously into the tile.
    text = (FIXTURES / "triton_masked_cp_async.ttgir.mlir").read_text()
    text = text.replace("arith.constant 8 : i32", "arith.constant 0 : i32")
    if not waited:
        text = text.replace(
            "    %wait = ttg.async_wait %group {num = 0 : i32}\n", ""
        ).replace("ttg.local_load %dst token %wait", "ttg.local_load %dst")
    fixture = tmp_path / "fully-masked-cp-async.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture(TRITON_PYTHON, "triton_capture.py", fixture, "fully-masked-cp-async")
    issue = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.cp_async.issue"
    )
    assert issue["source_region"] == "region:global:cp-async-zfill-no-read:0:64"
    assert issue["lane_effects"]["lanes"] == list(range(16))
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == verdict


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize(
    ("fixture", "source", "expected"),
    [
        ("triton_mbarrier.ttgir.mlir", "live-triton-mbarrier", [
            "triton.mbarrier.init",
            "triton.barrier.arrive",
            "triton.barrier.observe",
            "triton.mbarrier.arrive",
            "triton.mbarrier.wait_parity",
        ]),
        ("triton_cp_async.ttgir.mlir", "live-triton-cp-async", [
            "triton.cp_async.issue",
            "triton.cp_async.commit_group",
            "triton.cp_async.wait_group",
            "triton.barrier.arrive",
            "triton.barrier.observe",
            "triton.mem.read",
        ]),
    ],
)
def test_live_triton_capture_has_complete_v2_certificate_with_maximal(
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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_memdesc_subslice_requires_exact_byte_offset_evidence(tmp_path: Path):
    fixture = tmp_path / "memdesc-subslice.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1, 1], threadsPerWarp = [8, 4], warpsPerCTA = [4, 1], order = [1, 0]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 32, transposed = true, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @memdesc_subslice() {
    %dst = ttg.local_alloc : () -> !ttg.memdesc<128x16xbf16, #shared, #smem, mutable>
    %view = ttg.memdesc_subslice %dst[64, 0] : !ttg.memdesc<128x16xbf16, #shared, #smem, mutable> -> !ttg.memdesc<64x16xbf16, #shared, #smem, 128x16>
    %loaded = ttg.local_load %view : !ttg.memdesc<64x16xbf16, #shared, #smem, 128x16> -> tensor<64x16xbf16, #blocked>
    tt.return
  }
}
""")
    missing = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "memdesc-subslice-missing-evidence",
    )
    assert missing.returncode == 2
    diagnostic = json.loads(missing.stdout)["diagnostics"][0]
    assert diagnostic["code"] == "TRITON_BINDING_EVIDENCE_REQUIRED"
    operation_index = diagnostic["details"]["operation_index"]

    evidence = tmp_path / "memdesc-subslice.binding.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {str(operation_index): {"memdesc_byte_offset": 2048}},
    }))
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "memdesc-subslice",
        binding_evidence=evidence,
    )
    assert snapshot["operations"][0]["region"].endswith(":2048:2048")


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
        "triton.barrier.arrive",
        "triton.barrier.observe",
        "triton.mem.read",
    ]
    assert snapshot["operations"][0]["region"] == snapshot["operations"][-1]["region"]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "DATA_HAZARD"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_local_store_emits_shared_write(tmp_path: Path):
    fixture = tmp_path / "local-store.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @local_store() {
    %payload = arith.constant dense<0> : tensor<16xi8, #blocked>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    ttg.local_store %payload, %dst : tensor<16xi8, #blocked> -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %loaded = ttg.local_load %dst : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "live-triton-local-store",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mem.write",
        "triton.barrier.arrive",
        "triton.barrier.observe",
        "triton.mem.read",
    ]
    assert snapshot["operations"][0]["region"] == snapshot["operations"][-1]["region"]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "DATA_HAZARD"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_while_preserves_loop_carried_memdesc_provenance(tmp_path: Path):
    fixture = tmp_path / "loop-carried-memdesc.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @loop_carried_memdesc() {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %payload = arith.constant dense<0> : tensor<16xi8, #blocked>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %result:2 = scf.while (%before = %c0, %before_dst = %dst) : (index, !ttg.memdesc<16xi8, #shared, #smem, mutable>) -> (index, !ttg.memdesc<16xi8, #shared, #smem, mutable>) {
      %condition = arith.cmpi slt, %before, %c1 : index
      scf.condition(%condition) %before, %before_dst : index, !ttg.memdesc<16xi8, #shared, #smem, mutable>
    } do {
    ^bb0(%after: index, %after_dst: !ttg.memdesc<16xi8, #shared, #smem, mutable>):
      ttg.local_store %payload, %after_dst : tensor<16xi8, #blocked> -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
      %next = arith.addi %after, %c1 : index
      scf.yield %next, %after_dst : index, !ttg.memdesc<16xi8, #shared, #smem, mutable>
    }
    %loaded = ttg.local_load %result#1 : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "live-triton-loop-carried-memdesc",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.mem.write",
        "triton.barrier.arrive",
        "triton.barrier.observe",
        "triton.mem.read",
    ]
    assert snapshot["operations"][0]["region"] == snapshot["operations"][-1]["region"]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "DATA_HAZARD"


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
    assert check_module(module, engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_bitcast_and_poison_are_protocol_pure(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "",
        """    %poison = ub.poison : i32
    %bits = arith.constant 1065353216 : i32
    %high = tt.mulhiui %bits, %bits : i32
    %value = tt.bitcast %bits : i32 -> f32
    %external = tt.extern_elementwise %value {libname = "", libpath = "", pure = true, symbol = "__nv_asinf"} : (f32) -> f32
    %converted = tt.fp_to_fp %external, rounding = rtne : f32 -> f16""",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "bitcast-poison-pure",
    )
    assert snapshot["operations"] == []
    assert snapshot["extraction"]["classified_nodes"] == snapshot["extraction"]["executable_nodes_seen"]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_pointer_bitcast_preserves_equal_width_provenance(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%ptr: !tt.ptr<f32>",
        """    %cast = tt.bitcast %ptr : !tt.ptr<f32> -> !tt.ptr<i32>
    %value = tt.load %cast : !tt.ptr<i32>""",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "pointer-bitcast",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == ["triton.mem.read"]
    region = snapshot["regions"][0]
    assert region["allocation_id"] == "global-arg:0"
    assert region["extent"] == 4


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_pointer_broadcast_expands_provenance_shape(tmp_path: Path):
    fixture = tmp_path / "pointer-broadcast.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1, 8], threadsPerWarp = [32, 1], warpsPerCTA = [4, 1], order = [1, 0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @pointer_broadcast(%ptr: !tt.ptr<i8>) {
    %base = tt.splat %ptr : !tt.ptr<i8> -> tensor<128x1x!tt.ptr<i8>, #blocked>
    %broadcast = tt.broadcast %base : tensor<128x1x!tt.ptr<i8>, #blocked> -> tensor<128x8x!tt.ptr<i8>, #blocked>
    %value = tt.load %broadcast : tensor<128x8x!tt.ptr<i8>, #blocked>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "pointer-broadcast",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == ["triton.mem.read"]
    assert snapshot["regions"][0]["extent"] == 1


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_if_preserves_selected_pointer_provenance(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%ptr: !tt.ptr<i32>",
        """    %condition = arith.constant true
    %selected = scf.if %condition -> (!tt.ptr<i32>) {
      scf.yield %ptr : !tt.ptr<i32>
    } else {
      scf.yield %ptr : !tt.ptr<i32>
    }
    %value = tt.load %selected : !tt.ptr<i32>""",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "static-if-pointer",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == ["triton.mem.read"]
    assert snapshot["regions"][0]["allocation_id"] == "global-arg:0"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_cf_selects_only_reachable_block(tmp_path: Path):
    fixture = tmp_path / "static-cfg.ttgir.mlir"
    fixture.write_text("""\
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @static_cfg(%ptr: !tt.ptr<i32>, %limit: i32) {
    %pid = tt.get_program_id x : i32
    %done = arith.cmpi sge, %pid, %limit : i32
    cf.cond_br %done, ^bb1, ^bb2
  ^bb1:
    tt.return
  ^bb2:
    %value = arith.constant 1 : i32
    tt.store %ptr, %value : !tt.ptr<i32>
    tt.return
  }
}
""")
    evidence = tmp_path / "static-cfg.binding.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "pointer_allocations": {"0": "global:out"},
        "scalar_values": {"1": 1},
        "array_attributes": {},
        "unit_attributes": {},
    }))
    _output, active = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "static-cfg-active",
        grid_ctas=None,
        launch_grid=(3, 1, 1),
        binding_evidence=evidence,
    )
    assert [operation["op"] for operation in active["operations"]] == ["triton.mem.write"]
    _output, exited = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "static-cfg-exit",
        grid_ctas=None,
        launch_grid=(3, 1, 1),
        selected_cta=(2, 0, 0),
        binding_evidence=evidence,
    )
    assert exited["operations"] == []


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_for_does_not_treat_predicate_tensor_as_register_storage(tmp_path: Path):
    fixture = tmp_path / "predicate-loop-carried.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @predicate_loop_carried() {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %true = arith.constant true
    %mask = tt.splat %true : i1 -> tensor<128xi1, #blocked>
    %result = scf.for %i = %zero to %one step %one iter_args(%carried = %mask) -> tensor<128xi1, #blocked> {
      scf.yield %carried : tensor<128xi1, #blocked>
    }
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "predicate-loop-carried",
    )
    assert snapshot["operations"] == []


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_static_for_preserves_cp_async_token_provenance(tmp_path: Path):
    fixture = tmp_path / "cp-async-loop-carried.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#shared = #ttg.swizzled_shared<{vec = 1, perPhase = 1, maxPhase = 1, order = [0]}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @async_loop_carried(%base: !tt.ptr<i32>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %base_vector = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>, #blocked>
    %src = tt.addptr %base_vector, %range : tensor<16x!tt.ptr<i32>, #blocked>, tensor<16xi32, #blocked>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<16xi32, #shared, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %src, %dst : tensor<16x!tt.ptr<i32>, #blocked> -> <16xi32, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    %carried = scf.for %i = %zero to %one step %one iter_args(%token = %group) -> (!ttg.async.token) {
      scf.yield %token : !ttg.async.token
    }
    %wait = ttg.async_wait %carried {num = 0 : i32}
    %loaded = ttg.local_load %dst token %wait : !ttg.memdesc<16xi32, #shared, #smem, mutable> -> tensor<16xi32, #blocked>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "cp-async-loop-carried",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == [
        "triton.cp_async.issue",
        "triton.cp_async.commit_group",
        "triton.cp_async.wait_group",
        "triton.barrier.arrive",
        "triton.barrier.observe",
        "triton.mem.read",
    ]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_unknown_effect_is_not_misclassified_as_empty_protocol(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "",
        '    %true = arith.constant true\n    tt.assert %true, "unknown" : i1',
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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_cross_warp_convert_layout_orders_global_war(tmp_path: Path):
    fixture = tmp_path / "convert-layout-inplace.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
#blocked1 = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @convert_inplace(%base: !tt.ptr<f32>) {
    %range = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked>
    %bases = tt.splat %base : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked>
    %pointers = tt.addptr %bases, %range : tensor<512x!tt.ptr<f32>, #blocked>, tensor<512xi32, #blocked>
    %loaded = tt.load %pointers : tensor<512x!tt.ptr<f32>, #blocked>
    %converted = ttg.convert_layout %loaded : tensor<512xf32, #blocked> -> tensor<512xf32, #blocked1>
    %range1 = tt.make_range {end = 512 : i32, start = 0 : i32} : tensor<512xi32, #blocked1>
    %bases1 = tt.splat %base : !tt.ptr<f32> -> tensor<512x!tt.ptr<f32>, #blocked1>
    %pointers1 = tt.addptr %bases1, %range1 : tensor<512x!tt.ptr<f32>, #blocked1>, tensor<512xi32, #blocked1>
    tt.store %pointers1, %converted : tensor<512x!tt.ptr<f32>, #blocked1>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "convert-layout-inplace",
    )
    ops = [operation["op"] for operation in snapshot["operations"]]
    assert ops == [
        "triton.mem.read",
        "triton.barrier.arrive",
        "triton.barrier.observe",
        "triton.mem.write",
    ]
    barrier = snapshot["resources"][0]
    assert barrier["kind"] == "cta_barrier"
    assert barrier["visibility_mask"]["global_read"] is True
    assert barrier["visibility_mask"]["global_write"] is True
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_loop_induction_resolves_pointer_splat_per_activation(tmp_path: Path):
    fixture = tmp_path / "loop-pointer-splat.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [4], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @loop_pointer_splat(%out: !tt.ptr<f32>, %value: tensor<4xf32, #blocked>) {
    %zero = arith.constant 0 : index
    %four = arith.constant 4 : index
    %twelve = arith.constant 12 : index
    %offsets = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32, #blocked>
    scf.for %block_start = %zero to %twelve step %four {
      %block_start_i32 = arith.index_cast %block_start : index to i32
      %base = tt.addptr %out, %block_start_i32 : !tt.ptr<f32>, i32
      %bases = tt.splat %base : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>, #blocked>
      %pointers = tt.addptr %bases, %offsets : tensor<4x!tt.ptr<f32>, #blocked>, tensor<4xi32, #blocked>
      tt.store %pointers, %value : tensor<4x!tt.ptr<f32>, #blocked>
    }
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "loop-pointer-splat",
    )
    assert [operation["region"] for operation in snapshot["operations"]] == [
        "region:global:global-arg:0:0:16",
        "region:global:global-arg:0:16:16",
        "region:global:global-arg:0:32:16",
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
def test_triton_sha_bound_pointer_allocations_resolve_alias_identity(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%left: !tt.ptr<i32>, %right: !tt.ptr<i32>",
        """    %zero = arith.constant 0 : i32
    tt.store %left, %zero : !tt.ptr<i32>
    tt.store %right, %zero : !tt.ptr<i32>""",
    )
    evidence = tmp_path / "pointer-bindings.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {},
        "pointer_allocations": {"0": "global:left", "1": "global:right"},
    }))
    process = subprocess.run(
        [
            str(TRITON_PYTHON),
            str(ROOT / "tools" / "triton_capture.py"),
            str(fixture),
            "--source",
            "pointer-allocation-evidence",
            "--grid-ctas",
            "1",
            "--selected-cta",
            "0",
            "0",
            "0",
            "--binding-evidence",
            str(evidence),
        ],
        cwd=ROOT,
        capture_output=True,
    )
    process.check_returncode()
    snapshot = json.loads(process.stdout)
    assert [region["allocation_id"] for region in snapshot["regions"]] == [
        "global:left",
        "global:right",
    ]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_exact_float_index_arithmetic_folds(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%out: !tt.ptr<i32>",
        """    %seven = arith.constant 7 : i32
    %half = arith.constant 5.000000e-01 : f32
    %one = arith.constant 1.000000e+00 : f32
    %as_float = arith.sitofp %seven : i32 to f32
    %scaled = arith.mulf %as_float, %half : f32
    %shifted = arith.addf %scaled, %one : f32
    %floored = math.floor %shifted : f32
    %index = arith.fptosi %floored : f32 to i32
    %in_range = arith.cmpf olt, %floored, %as_float : f32
    %target = tt.addptr %out, %index : !tt.ptr<i32>, i32
    tt.store %target, %seven, %in_range : !tt.ptr<i32>""",
    )
    evidence = tmp_path / "pointer-bindings.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {},
        "pointer_allocations": {"0": "global:out"},
    }))
    process = subprocess.run(
        [
            str(TRITON_PYTHON),
            str(ROOT / "tools" / "triton_capture.py"),
            str(fixture),
            "--source",
            "float-index",
            "--grid-ctas",
            "1",
            "--selected-cta",
            "0",
            "0",
            "0",
            "--binding-evidence",
            str(evidence),
        ],
        cwd=ROOT,
        capture_output=True,
    )
    process.check_returncode()
    snapshot = json.loads(process.stdout)
    regions = {region["id"]: region for region in snapshot["regions"]}
    assert [
        (operation["op"], regions[operation["region"]]["offset"], regions[operation["region"]]["extent"])
        for operation in snapshot["operations"]
    ] == [("triton.mem.write", 16, 4)]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_null_test_of_an_evidenced_pointer_folds(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%out: !tt.ptr<i32>, %bias: !tt.ptr<i32>",
        """    %zero = arith.constant 0 : i32
    %null = arith.constant 0 : i64
    %address = tt.ptr_to_int %bias : !tt.ptr<i32> -> i64
    %has_bias = arith.cmpi ne, %address, %null : i64
    %no_bias = arith.cmpi eq, %address, %null : i64
    scf.if %has_bias {
      tt.store %bias, %zero : !tt.ptr<i32>
    }
    scf.if %no_bias {
      tt.store %out, %zero : !tt.ptr<i32>
    }""",
    )
    evidence = tmp_path / "pointer-bindings.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {},
        "pointer_allocations": {"0": "global:out", "1": "global:bias"},
    }))
    process = subprocess.run(
        [
            str(TRITON_PYTHON),
            str(ROOT / "tools" / "triton_capture.py"),
            str(fixture),
            "--source",
            "pointer-null-test",
            "--grid-ctas",
            "1",
            "--selected-cta",
            "0",
            "0",
            "0",
            "--binding-evidence",
            str(evidence),
        ],
        cwd=ROOT,
        capture_output=True,
    )
    process.check_returncode()
    snapshot = json.loads(process.stdout)
    assert [region["allocation_id"] for region in snapshot["regions"]] == ["global:bias"]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_sha_bound_runtime_scalar_resolves_native_ir_address(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%base: !tt.ptr<i32>, %offset: i32",
        """    %true = arith.constant true
    %one = arith.extui %true : i1 to i32
    %two = arith.constant 2 : i32
    %scaled = arith.divsi %offset, %two : i32
    %final = arith.addi %scaled, %one : i32
    %pointer = tt.addptr %base, %final : !tt.ptr<i32>, i32
    %value = tt.load %pointer : !tt.ptr<i32>""",
    )
    evidence = tmp_path / "scalar-binding.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {},
        "pointer_allocations": {"0": "global:data"},
        "scalar_values": {"1": 8},
    }))
    process = subprocess.run(
        [
            str(TRITON_PYTHON),
            str(ROOT / "tools" / "triton_capture.py"),
            str(fixture),
            "--source",
            "runtime-scalar-evidence",
            "--grid-ctas",
            "1",
            "--selected-cta",
            "0",
            "0",
            "0",
            "--binding-evidence",
            str(evidence),
        ],
        cwd=ROOT,
        capture_output=True,
    )
    process.check_returncode()
    snapshot = json.loads(process.stdout)
    assert snapshot["operations"][0]["region"] == "region:global:global:data:20:4"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_pointer_allocation_evidence_must_name_pointer_arguments(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%left: !tt.ptr<i32>, %right: !tt.ptr<i32>",
        """    %zero = arith.constant 0 : i32
    tt.store %left, %zero : !tt.ptr<i32>
    tt.store %right, %zero : !tt.ptr<i32>""",
    )
    evidence = tmp_path / "invalid-pointer-bindings.json"
    evidence.write_text(json.dumps({
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "unit_attributes": {},
        "array_attributes": {},
        "pointer_allocations": {"0": "global:left", "99": "global-arg:1"},
    }))
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "invalid-pointer-allocation-evidence",
        binding_evidence=evidence,
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == "TRITON_BINDING_EVIDENCE_INVALID"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_integer_evaluation_uses_fixed_width_mlir_semantics(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%base: !tt.ptr<i32>",
        """    %max = arith.constant 127 : i8
    %one8 = arith.constant 1 : i8
    %zero8 = arith.constant 0 : i8
    %wrapped = arith.addi %max, %one8 : i8
    %negative = arith.cmpi slt, %wrapped, %zero8 : i8
    %zero32 = arith.constant 0 : i32
    %one32 = arith.constant 1 : i32
    %branch_offset = arith.select %negative, %zero32, %one32 : i32
    %minus_one = arith.constant -1 : i8
    %two8 = arith.constant 2 : i8
    %unsigned_quotient = arith.divui %minus_one, %two8 : i8
    %minus_three = arith.constant -3 : i8
    %signed_remainder = arith.remsi %minus_three, %two8 : i8
    %ceiling = arith.ceildivsi %minus_three, %two8 : i8
    %combined = arith.addi %unsigned_quotient, %signed_remainder : i8
    %combined_with_ceiling = arith.addi %combined, %ceiling : i8
    %combined32 = arith.extsi %combined_with_ceiling : i8 to i32
    %offset = arith.addi %branch_offset, %combined32 : i32
    %pointer = tt.addptr %base, %offset : !tt.ptr<i32>, i32
    %value = tt.load %pointer : !tt.ptr<i32>""",
    )
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "fixed-width-integer-semantics",
    )
    assert snapshot["operations"][0]["region"] == "region:global:global-arg:0:500:4"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_loop_does_not_reuse_stale_activation_value(tmp_path: Path):
    fixture = _scalar_pointer_ttgir(
        tmp_path,
        "%base: !tt.ptr<i32>",
        """    %lower = arith.constant 0 : index
    %upper = arith.constant 2 : index
    %step = arith.constant 1 : index
    %one = arith.constant 1 : i32
    %four = arith.constant 4 : i32
    scf.for %iteration = %lower to %upper step %step {
      %iteration_i32 = arith.index_cast %iteration : index to i32
      %denominator = arith.subi %one, %iteration_i32 : i32
      %offset = arith.divsi %four, %denominator : i32
      %pointer = tt.addptr %base, %offset : !tt.ptr<i32>, i32
      %value = tt.load %pointer : !tt.ptr<i32>
    }""",
    )
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "stale-loop-activation-value",
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == "TRITON_MEMORY_ACCESS_UNRESOLVED"


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
def test_triton_boolean_pointer_accessed_through_byte_bitcast(tmp_path: Path):
    fixture = tmp_path / "boolean-pointer.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @boolean_store(%base: !tt.ptr<i1>) {
    %range = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32, #blocked>
    %splat = tt.splat %base : !tt.ptr<i1> -> tensor<128x!tt.ptr<i1>, #blocked>
    %flags = tt.addptr %splat, %range : tensor<128x!tt.ptr<i1>, #blocked>, tensor<128xi32, #blocked>
    %bytes = tt.bitcast %flags : tensor<128x!tt.ptr<i1>, #blocked> -> tensor<128x!tt.ptr<i8>, #blocked>
    %one = arith.constant dense<1> : tensor<128xi8, #blocked>
    tt.store %bytes, %one : tensor<128x!tt.ptr<i8>, #blocked>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "boolean-pointer",
    )
    assert [(operation["op"], operation["region"]) for operation in snapshot["operations"]] == [
        ("triton.mem.write", "region:global:global-arg:0:0:128"),
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
    load, store = snapshot["operations"]
    assert len(load["lane_effects"]["lanes"]) == 128
    assert len(store["lane_effects"]["lanes"]) == 16
    assert load["execution_evidence"]["redundant_policy"] == "LOAD_ALL"
    assert store["execution_evidence"]["redundant_policy"] == "STORE_REPRESENTATIVE"
    assert snapshot["extraction"]["elided_nodes"] == 4
    module = triton_snapshot_to_module(snapshot)
    assert module.logical_stream_count == 128
    assert check_module(module, engine="maximal")["verdict"] == "DATA_HAZARD"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_lane_indexed_store_detects_repeated_address_collision(tmp_path: Path):
    fixture = tmp_path / "repeated-address-store.ttgir.mlir"
    fixture.write_text("""\
#blocked = #ttg.blocked<{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}>
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @repeated_address_store(%base: !tt.ptr<i32>) {
    %pointers = tt.splat %base : !tt.ptr<i32> -> tensor<128x!tt.ptr<i32>, #blocked>
    %zero = arith.constant dense<0> : tensor<128xi32, #blocked>
    tt.store %pointers, %zero : tensor<128x!tt.ptr<i32>, #blocked>
    tt.return
  }
}
""")
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "repeated-address-store",
    )
    operation = snapshot["operations"][0]
    assert len(operation["lane_effects"]["lanes"]) == 128
    assert set(operation["lane_effects"]["regions"]) == {
        "region:global:global-arg:0:0:4"
    }

    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_lane_indexed_disjoint_loads_remain_safe():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_disjoint_memory.ttgir.mlir",
        "lane-indexed-disjoint-load",
    )
    assert sum(
        len(operation["lane_effects"]["lanes"])
        for operation in snapshot["operations"]
    ) == 128
    assert all(operation["op"] == "triton.mem.read" for operation in snapshot["operations"])
    module = triton_snapshot_to_module(snapshot)
    assert module.logical_stream_count == 128
    assert check_module(module, engine="maximal")["verdict"] == "SAFE"


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
def test_triton_capture_selects_first_cta_from_exact_launch_grid():
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_memory.ttgir.mlir",
        "multi-cta-selected-first",
        grid_ctas=6,
        launch_grid=(2, 3),
    )
    assert snapshot["target"]["grid_scope"] == "single_cta"
    assert snapshot["target"]["launch_grid"] == [2, 3, 1]
    assert snapshot["target"]["selected_cta"] == [0, 0, 0]
    module = triton_snapshot_to_module(snapshot)
    result = check_module(module)
    assert result["verdict"] == "DATA_HAZARD"
    assert module.raw["target"]["selected_cta"] == [0, 0, 0]
    assert module.raw["target"]["launch_grid"] == [2, 3, 1]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_capture_evaluates_program_id_and_num_programs_for_selected_cta(tmp_path: Path):
    module = (FIXTURES / "triton_static_memory.ttgir.mlir").read_text()
    path = tmp_path / "selected-grid.ttgir.mlir"
    path.write_text(module.replace(
        "    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>",
        """    %pid = tt.get_program_id x : i32
    %programs = tt.get_num_programs y : i32
    %origin = arith.addi %pid, %programs : i32
    %origin_vector = tt.splat %origin : i32 -> tensor<16xi32, #blocked>
    %local_range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %range = arith.addi %local_range, %origin_vector : tensor<16xi32, #blocked>""",
    ))
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        path,
        "selected-grid-values",
        grid_ctas=6,
        launch_grid=(2, 3),
        selected_cta=(1, 2, 0),
    )
    assert snapshot["target"]["selected_cta"] == [1, 2, 0]
    assert [operation["region"] for operation in snapshot["operations"]] == [
        "region:global:global-arg:0:4:16",
        "region:global:global-arg:0:4:16",
    ]


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_capture_rejects_inconsistent_launch_grid_evidence():
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_static_memory.ttgir.mlir",
        "inconsistent-grid",
        grid_ctas=5,
        launch_grid=(2, 3),
    )
    assert process.returncode == 2
    result = json.loads(process.stdout)
    assert result["diagnostics"][0]["code"] == "TRITON_GRID_EVIDENCE_INVALID"


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
    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
    assert snapshot["target"]["cta_threads"] == 256
    assert [(actor["id"], actor["kind"], len(actor["participants"]))
            for actor in snapshot["actors"]] == [
        ("cta", "WARPGROUP", 128),
        ("partition:0", "WARPGROUP", 128),
    ]
    assert [(operation["actor"], operation["op"])
            for operation in snapshot["operations"]] == [
        ("cta", "triton.mem.write"),
        ("cta", "triton.barrier.arrive"),
        ("cta", "triton.barrier.observe"),
        ("cta", "triton.mem.read"),
        ("partition:0", "triton.mem.read"),
        ("cta", "triton.mem.read"),
    ]
    assert snapshot["operations"][4]["depends_on"] == [snapshot["operations"][0]["id"]]
    assert snapshot["operations"][5]["depends_on"] == [snapshot["operations"][4]["id"]]
    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
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
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_static_blocks_preserve_predicates_init_and_strided_offsets():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_static_block_device.tvm.json",
        "live-tilelang-static-block",
    )
    assert [
        region
        for operation in snapshot["operations"]
        for region in _compact_field_values(operation, "region")
    ] == [
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
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize(("variant", "verdict"), [("same_thread", "SAFE"), ("cross_thread", "DATA_HAZARD")])
def test_tilelang_dtype_changing_view_uses_its_own_element_width(variant: str, verdict: str):
    # Lowered with thread storage sync disabled: each thread writes 16
    # bytes through the float16 tile and reads them back through the int8
    # T.view, either its own bytes or another thread's (reversed rows).
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        f"tilelang_view_alias_{variant}_device.tvm.json",
        f"live-tilelang-view-alias-{variant}",
    )
    extents = {
        int(region.split(":")[-1])
        for operation in snapshot["operations"]
        for region in _compact_field_values(operation, "region")
        if region.startswith("region:shared:")
    }
    assert extents == {16}
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == verdict


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_static_while_unfolds_bind_updates_to_exact_regions():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_static_while_device.tvm.json",
        "live-tilelang-static-while",
    )
    assert [
        region
        for operation in snapshot["operations"]
        for region in _compact_field_values(operation, "region")
    ] == [
        "region:global:destination:4:4",
        "region:global:destination:36:4",
        "region:global:destination:68:4",
    ]
    assert snapshot["extraction"]["static_expansions"] == 3
    assert snapshot["extraction"]["elided_nodes"] == 4
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


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
@pytest.mark.parametrize(
    ("position", "verdict"),
    [("after_write", "SAFE"), ("before_write", "DATA_HAZARD")],
)
def test_tilelang_proxy_fence_publishes_only_earlier_generic_writes(position: str, verdict: str):
    # Every thread fences, then a CTA barrier, then the elected TMA store
    # (tilelang_tma_store_fence_placement.py); each lane's fence binds to that
    # store at its own program position.
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        f"tilelang_tma_store_fence_{position}_device.tvm.json",
        f"live-tilelang-tma-store-fence-{position}",
    )
    operations = [operation["op"] for operation in snapshot["operations"]]
    fence_index = operations.index("tilelang.proxy_fence")
    write_index = operations.index("tilelang.mem.write")
    assert (fence_index < write_index) == (position == "before_write")
    fence = snapshot["operations"][fence_index]
    assert fence["instances"] == list(range(128))
    assert fence["fields"]["to_domain"] == "TMA"
    assert fence["fields"]["visibility_mask"]["shared"] is True
    assert fence["fields"]["visibility_mask"]["global_write"] is False
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == verdict


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize(("name", "verdict"), [("disjoint", "SAFE"), ("overlap", "DATA_HAZARD")])
def test_tilelang_cross_thread_cp_async_waw_needs_order_between_copies(name: str, verdict: str):
    # Each thread waits its own group and a CTA barrier follows, but nothing
    # orders thread 0's copy against thread 1's (tilelang_cp_async_cross_thread_waw_kernel.py).
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        f"tilelang_cp_async_cross_thread_{name}_device.tvm.json",
        f"live-tilelang-cp-async-cross-thread-{name}",
    )
    issue = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "tilelang.cp_async.issue"
    )
    assert issue["instances"] == [0, 1]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == verdict


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
def test_tilelang_tma_box_past_tensor_end_zero_fills_without_global_read(tmp_path: Path):
    # A has 48 rows but the 64-row box still lands in shared: rows 48..63 are
    # zero-filled from a private source, never read from A.
    evidence = json.loads(
        (FIXTURES / "tilelang_warp_specialized_descriptors.json").read_text()
    )
    evidence["descriptors"]["A_desc"]["shape"] = [48, 16]
    path = tmp_path / "descriptors.json"
    path.write_text(json.dumps(evidence))
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_warp_specialized_device.tvm.json",
        "live-tilelang-tma-zero-fill",
        descriptor_evidence=str(path),
    )
    regions = {region["id"]: region for region in snapshot["regions"]}
    a_sources = [
        (regions[transfer["source_region"]], transfer["bytes"])
        for operation in snapshot["operations"]
        if operation["op"] == "tilelang.tma_load.issue"
        for transfers in _compact_field_values(operation, "transfers")
        for transfer in transfers
        if regions[transfer["source_region"]]["allocation_id"] in {"global:A", "tma-zero-fill:A_desc:0"}
    ]
    assert sorted(
        (source["allocation_id"], source["offset"], source["extent"], count)
        for source, count in a_sources
    ) == [
        ("global:A", 0, 48 * 16 * 2, 48 * 16 * 2),
        ("tma-zero-fill:A_desc:0", 48 * 16 * 2, 16 * 16 * 2, 16 * 16 * 2),
    ]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_guarded_load_reads_only_on_lanes_whose_guard_holds():
    # B[tx] = if_then_else(1 <= tx, A[tx - 1], 0): lane 0 never reads A[-1]
    # (tilelang_capture_support_kernels.py).
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_guarded_load_device.tvm.json",
        "live-tilelang-guarded-load",
    )
    regions = {region["id"]: region for region in snapshot["regions"]}
    reads = [
        (instance, regions[region]["offset"])
        for operation in snapshot["operations"]
        if operation["op"] == "tilelang.mem.read"
        for instance, region in zip(operation["instances"], _compact_field_values(operation, "region"))
        if regions[region]["allocation_id"].endswith("A")
    ]
    assert sorted(reads) == [(lane, (lane - 1) * 4) for lane in range(1, 128)]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_descriptor_tma_store_drops_rows_past_tensor_end():
    # A 64-row box stored to a 48-row tensor writes only rows 0..47; the
    # swizzled shared source is then read as a whole tile.
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_descriptor_tma_store_device.tvm.json",
        "live-tilelang-descriptor-tma-store",
        descriptor_evidence="tilelang_descriptor_tma_store_descriptors.json",
    )
    regions = {region["id"]: region for region in snapshot["regions"]}
    stores = [
        (
            regions[operation["fields"]["source_region"]]["extent"],
            regions[operation["fields"]["destination_region"]]["allocation_id"],
            regions[operation["fields"]["destination_region"]]["offset"],
            regions[operation["fields"]["destination_region"]]["extent"],
            operation["fields"]["bytes"],
        )
        for operation in snapshot["operations"]
        if operation["op"] == "tilelang.tma_store.issue"
    ]
    assert stores == [(64 * 64 * 2, "global:D", 0, 48 * 64 * 2, 48 * 64 * 2)]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize(
    ("selected_cta", "tile"),
    [((0, 0, 0), (0, 0)), ((1, 0, 0), (0, 1)), ((3, 1, 0), (3, 1)), ((0, 2, 0), (3, 2))],
)
def test_tilelang_threadblock_swizzle_rebinds_the_selected_cta(selected_cta, tile):
    # rasterization2DRow<2> on a 4x4 grid: CTA (x, y) writes tile
    # (column, row) of C (tilelang_capture_support_kernels.py).
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_swizzled_blocks_device.tvm.json",
        "live-tilelang-swizzled-blocks",
        launch_grid=(4, 4, 1),
        selected_cta=selected_cta,
    )
    regions = {region["id"]: region for region in snapshot["regions"]}
    offsets = [
        regions[region]["offset"]
        for operation in snapshot["operations"]
        if operation["op"] == "tilelang.mem.write"
        for region in _compact_field_values(operation, "region")
    ]
    column, row = tile
    assert min(offsets) == (row * 16 * 64 + column * 16) * 4
    assert max(offsets) == ((row * 16 + 15) * 64 + column * 16 + 14) * 4


def _lane_regions(snapshot: dict, op: str, field: str = "region") -> list[tuple[int, dict]]:
    regions = {region["id"]: region for region in snapshot["regions"]}
    return [
        (instance, regions[region])
        for operation in snapshot["operations"]
        if operation["op"] == op
        for instance, region in zip(operation["instances"], _compact_field_values(operation, field))
    ]


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_full_warp_shuffle_is_a_register_exchange():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_warp_shuffle_device.tvm.json",
        "live-tilelang-warp-shuffle",
    )
    reads = [
        (lane, region["offset"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.read")
        if region["address_space"] == "GLOBAL"
    ]
    assert sorted(reads) == [(lane, lane * 4) for lane in range(128)]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_false_cp_async_predicate_zero_fills_without_source_read():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_predicated_cp_async_device.tvm.json",
        "live-tilelang-predicated-cp-async",
    )
    sources = sorted(
        (lane, region["allocation_id"], region["offset"])
        for lane, region in _lane_regions(snapshot, "tilelang.cp_async.issue", "source_region")
    )
    assert sources[0] == (0, "cp-async-zero-fill", 0)
    assert [offset for _lane, _allocation, offset in sources[1:]] == [
        (lane - 1) * 4 for lane in range(1, 128)
    ]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_ldmatrix_reads_each_lane_supplied_row():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_ldmatrix_device.tvm.json",
        "live-tilelang-ldmatrix",
    )
    rows = [
        (lane, region["offset"], region["extent"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.read")
        if region["address_space"] == "SHARED"
    ]
    assert sorted(rows) == [(lane, lane * 16, 16) for lane in range(32)]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


def _wgmma_proof_operand(
    offset_units: int,
    *,
    slot: str = "A_SHARED",
    stage_region: str = "region:shared:test:0:4096",
    descriptor_id: str = "warpgroup:00:desc_a",
    k_major: bool = True,
):
    return _wgmma_issue_operand(
        slot=slot,
        shape="m64n64k16",
        dtype="fp16",
        k_major=k_major,
        descriptor_offset_units=offset_units,
        descriptor={
            "descriptor_id": descriptor_id,
            "layout_type": 2 if slot == "A_SHARED" else 1,
            "leading_byte_offset": 1 if slot == "A_SHARED" else 0,
            "stride_byte_offset": 32 if slot == "A_SHARED" else 64,
            "selected_stage": {
                "region_id": stage_region,
                "allocation_id": "shared:test",
                "address_space": "SHARED",
                "offset": 0,
                "extent": 4096,
            },
        },
    )


def test_wgmma_exact_group_union_proves_complete_k16_partition():
    issues = [
        {
            "issue_template_id": f"issue:{index}",
            "engine_seq": index,
            "operands": [
                _wgmma_proof_operand(index * 2),
                _wgmma_proof_operand(
                    index * 128,
                    slot="B_SHARED",
                    descriptor_id="warpgroup:00:desc_b",
                    k_major=False,
                ),
            ],
        }
        for index in range(2)
    ]
    proof = _prove_wgmma_group(
        "warpgroup:00", "wgmma:warpgroup:00", 0, "commit:0", issues
    )
    assert proof["status"] == "PROVED"
    assert proof["engine_seqs"] == [0, 1]
    assert all(
        operand["region_precision"] == "EXACT_GROUP_UNION"
        and operand["equal"] is True
        and operand["expected_logical_union"] == [{"offset": 0, "extent": 4096}]
        for operand in proof["operands"]
    )


def test_wgmma_exact_group_union_rejects_duplicate_slice_and_mixed_stage():
    duplicate = [
        {
            "issue_template_id": f"issue:{index}",
            "engine_seq": index,
            "operands": [_wgmma_proof_operand(0)],
        }
        for index in range(2)
    ]
    with pytest.raises(TileLangCaptureError) as duplicate_error:
        _prove_wgmma_group(
            "warpgroup:00", "wgmma:warpgroup:00", 0, "commit:0", duplicate
        )
    assert duplicate_error.value.code == "TILELANG_WGMMA_COMMITTED_GROUP_REGION_MISMATCH"

    mixed = [
        {
            "issue_template_id": "issue:0",
            "engine_seq": 0,
            "operands": [_wgmma_proof_operand(0)],
        },
        {
            "issue_template_id": "issue:1",
            "engine_seq": 1,
            "operands": [_wgmma_proof_operand(
                2,
                stage_region="region:shared:test:4096:4096",
            )],
        },
    ]
    with pytest.raises(TileLangCaptureError) as mixed_error:
        _prove_wgmma_group(
            "warpgroup:00", "wgmma:warpgroup:00", 0, "commit:0", mixed
        )
    assert mixed_error.value.code == "TILELANG_WGMMA_COMMITTED_GROUP_STAGE_MIXED"


def test_wgmma_exact_group_union_rejects_unknown_profile_and_layout():
    descriptor = {
        "descriptor_id": "warpgroup:00:desc",
        "layout_type": 7,
        "leading_byte_offset": 0,
        "stride_byte_offset": 0,
        "selected_stage": {
            "region_id": "region:shared:test:0:4096",
            "allocation_id": "shared:test",
            "address_space": "SHARED",
            "offset": 0,
            "extent": 4096,
        },
    }
    with pytest.raises(TileLangCaptureError) as shape_error:
        _wgmma_issue_operand(
            slot="A_SHARED",
            shape="m64n16k16",
            dtype="fp16",
            k_major=True,
            descriptor_offset_units=0,
            descriptor=descriptor,
        )
    assert shape_error.value.code == "TILELANG_WGMMA_ISSUE_FOOTPRINT_UNRESOLVED"

    with pytest.raises(TileLangCaptureError) as layout_error:
        _wgmma_issue_operand(
            slot="A_SHARED",
            shape="m64n64k16",
            dtype="fp16",
            k_major=True,
            descriptor_offset_units=0,
            descriptor=descriptor,
        )
    assert layout_error.value.code == "TILELANG_WGMMA_DESCRIPTOR_LAYOUT_UNSUPPORTED"


def test_wgmma_layout_hull_bounds_non_tma_descriptor_reads():
    def hull(**overrides):
        arguments = {
            "mn": 64, "k": 16, "element_bytes": 2, "k_major": True,
            "layout_type": 2, "leading": 1, "stride": 32, "start": 0,
        }
        return _wgmma_layout_hull(**{**arguments, **overrides})

    # K-major 64B swizzle: every K16 slice stays inside the 8 x 512B panel.
    assert hull() == (0, 4096)
    assert hull(start=32) == (0, 4096)
    assert hull(start=4096 + 32) == (4096, 8192)
    # MN-major 128B swizzle with one MN atom: K16 spans two 1KiB atoms,
    # whichever of LBO/SBO carries the K stride.
    assert hull(k_major=False, layout_type=1, leading=0, stride=64, start=2048) == (2048, 4096)
    assert hull(k_major=False, layout_type=1, leading=64, stride=0, start=2048) == (2048, 4096)
    # Interleaved core matrices are bounded chunk by chunk.
    assert hull(layout_type=0, leading=8, stride=16) == (0, 2048)
    assert hull(layout_type=5) is None
    assert hull(stride=-1) is None

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
    assert all(
        operation["execution_scope"] == "COLLECTIVE"
        and len(operation["instances"]) == 128
        for operation in snapshot["operations"]
        if operation["op"].startswith("tilelang.wgmma.")
    )
    wgmma_issues = [
        operation for operation in snapshot["operations"]
        if operation["op"] == "tilelang.wgmma.issue"
    ]
    for owner in {operation["owner"] for operation in wgmma_issues}:
        owner_issues = [
            operation for operation in wgmma_issues
            if operation["owner"] == owner
        ]
        assert all(
            set(_compact_field_values(operation, "engine_id")) == {owner}
            for operation in owner_issues
        )
        assert [
            _compact_field_values(operation, "engine_seq")[0]
            for operation in owner_issues
        ] == list(range(len(owner_issues)))
    tma_issues = [
        operation for operation in snapshot["operations"]
        if operation["op"] == "tilelang.tma_load.issue"
    ]
    assert tma_issues
    assert all(
        issue["execution_scope"] == "ELECTED" and len(issue["instances"]) == 1
        for issue in tma_issues
    )
    assert all(_compact_field_values(issue, "transfers")[0] for issue in tma_issues)
    assert all(
        not _compact_field_values(issue, "source_region")
        and not _compact_field_values(issue, "destination_region")
        for issue in tma_issues
    )
    assert all(
        _compact_field_values(issue, "bytes")[0]
        == sum(
            transfer["bytes"]
            for transfer in _compact_field_values(issue, "transfers")[0]
        )
        for issue in tma_issues
    )
    regions = {region["id"]: region for region in snapshot["regions"]}
    issue = next(operation for operation in snapshot["operations"]
                 if operation["op"] == "tilelang.wgmma.issue")
    accumulator = regions[_compact_field_values(issue, "destination_region")[0]]
    assert any(
        operation["op"] == "tilelang.mem.read"
        and any(
            regions[region]["allocation_id"] == accumulator["allocation_id"]
            for region in _compact_field_values(operation, "region")
        )
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
    assert snapshot["extraction"]["operation_records"] == 600
    assert snapshot["extraction"]["logical_events_emitted"] == 600
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["representation"] == "parametric-protocol"
    assert result["statistics"]["materialized_event_objects"] == 0
    assert result["statistics"]["materialized_artifact_events"] == 0


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_cp_async_rejects_subword_per_thread_transfer(tmp_path: Path):
    text = (FIXTURES / "triton_cp_async.ttgir.mlir").read_text()
    text = text.replace("!tt.ptr<i32>", "!tt.ptr<i16>")
    text = text.replace("!ttg.memdesc<16xi32", "!ttg.memdesc<16xi16")
    text = text.replace("-> <16xi32, #shared", "-> <16xi16, #shared")
    text = text.replace(
        "mutable> -> tensor<16xi32, #blocked>",
        "mutable> -> tensor<16xi16, #blocked>",
    )
    fixture = tmp_path / "subword-cp-async.ttgir.mlir"
    fixture.write_text(text)
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "subword-cp-async",
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == (
        "TRITON_CP_ASYNC_TRANSFER_WIDTH_UNSUPPORTED"
    )


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize(("size_per_thread", "lane_bytes"), [(8, [16, 16]), (2, [4] * 8)])
def test_triton_cp_async_vectorizes_contiguous_subword_elements(
    tmp_path: Path,
    size_per_thread: int,
    lane_bytes: list[int],
):
    text = (FIXTURES / "triton_cp_async.ttgir.mlir").read_text()
    text = text.replace("sizePerThread = [1]", f"sizePerThread = [{size_per_thread}]")
    text = text.replace("!tt.ptr<i32>", "!tt.ptr<f16>")
    text = text.replace("!ttg.memdesc<16xi32", "!ttg.memdesc<16xf16")
    text = text.replace("-> <16xi32, #shared", "-> <16xf16, #shared")
    text = text.replace(
        "mutable> -> tensor<16xi32, #blocked>",
        "mutable> -> tensor<16xf16, #blocked>",
    )
    fixture = tmp_path / f"vector-cp-async-{size_per_thread}.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        f"vector-cp-async-{size_per_thread}",
    )
    issue = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.cp_async.issue"
    )
    assert issue["lane_effects"]["bytes"] == lane_bytes
    assert issue["lane_effects"]["lanes"] == list(range(len(lane_bytes)))
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


_TRANSPOSED_CP_ASYNC_TTGIR = """\
#blocked = #ttg.blocked<{sizePerThread = [8, 1], threadsPerWarp = [8, 4], warpsPerCTA = [1, 4], order = [0, 1]}>
#shared = #ttg.nvmma_shared<{swizzlingByteWidth = 128, transposed = TRANSPOSED, elementBitWidth = 16}>
#smem = #ttg.shared_memory
module attributes {"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32} {
  tt.func public @transposed_operand(%base: !tt.ptr<f16>) {
    %rows = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>>
    %rows_2d = tt.expand_dims %rows {axis = 1 : i32} : tensor<64xi32, #ttg.slice<{dim = 1, parent = #blocked}>> -> tensor<64x1xi32, #blocked>
    %cols = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>>
    %cols_2d = tt.expand_dims %cols {axis = 0 : i32} : tensor<16xi32, #ttg.slice<{dim = 0, parent = #blocked}>> -> tensor<1x16xi32, #blocked>
    %stride = arith.constant dense<128> : tensor<1x16xi32, #blocked>
    %col_offsets = arith.muli %cols_2d, %stride : tensor<1x16xi32, #blocked>
    %row_tile = tt.broadcast %rows_2d : tensor<64x1xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %col_tile = tt.broadcast %col_offsets : tensor<1x16xi32, #blocked> -> tensor<64x16xi32, #blocked>
    %offsets = arith.addi %row_tile, %col_tile : tensor<64x16xi32, #blocked>
    %base_tile = tt.splat %base : !tt.ptr<f16> -> tensor<64x16x!tt.ptr<f16>, #blocked>
    %src = tt.addptr %base_tile, %offsets : tensor<64x16x!tt.ptr<f16>, #blocked>, tensor<64x16xi32, #blocked>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %src, %dst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    %wait = ttg.async_wait %group {num = 0 : i32}
    %loaded = ttg.local_load %dst token %wait : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> -> tensor<64x16xf16, #blocked>
    tt.return
  }
}
"""


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_cp_async_follows_transposed_shared_order(tmp_path: Path):
    fixture = tmp_path / "transposed-cp-async.ttgir.mlir"
    fixture.write_text(_TRANSPOSED_CP_ASYNC_TTGIR.replace("TRANSPOSED", "true"))
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        fixture,
        "transposed-cp-async",
    )
    regions = {region["id"]: region for region in snapshot["regions"]}
    issue = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.cp_async.issue"
    )
    effects = issue["lane_effects"]
    # Lane (warp, thread) owns rows 8*(thread%8).. of column 4*warp +
    # thread//8: contiguous in the strided column-major source and in the
    # transposed (dim-0-fastest) tile.
    expected = [
        2 * (64 * (4 * (lane // 32) + lane % 32 // 8) + 8 * (lane % 8))
        for lane in range(128)
    ]
    assert effects["bytes"] == [16] * 128
    assert effects["lanes"] == list(range(128))
    assert [regions[region]["offset"] for region in effects["destination_regions"]] == expected
    read = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.mem.read"
    )
    assert [
        (regions[region]["offset"], regions[region]["extent"])
        for region in read["lane_effects"]["regions"]
    ] == [(offset, 16) for offset in expected]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"

    row_major = tmp_path / "row-major-cp-async.ttgir.mlir"
    row_major.write_text(
        _TRANSPOSED_CP_ASYNC_TTGIR.replace("TRANSPOSED", "false")
        .replace("swizzlingByteWidth = 128", "swizzlingByteWidth = 32")
    )
    process = _run_capture(TRITON_PYTHON, "triton_capture.py", row_major, "row-major-cp-async")
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == (
        "TRITON_CP_ASYNC_TRANSFER_WIDTH_UNSUPPORTED"
    )


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize(("waited", "verdict"), [(True, "SAFE"), (False, "DATA_HAZARD")])
def test_triton_dot_operand_load_reads_whole_tile(tmp_path: Path, waited: bool, verdict: str):
    text = _TRANSPOSED_CP_ASYNC_TTGIR.replace("TRANSPOSED", "true").replace(
        "#smem = #ttg.shared_memory",
        "#smem = #ttg.shared_memory\n"
        "#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>",
    ).replace(
        "-> tensor<64x16xf16, #blocked>",
        "-> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>",
    )
    if not waited:
        text = text.replace(
            "    %wait = ttg.async_wait %group {num = 0 : i32}\n", ""
        ).replace("%loaded = ttg.local_load %dst token %wait", "%loaded = ttg.local_load %dst")
    fixture = tmp_path / "dot-operand-load.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture(TRITON_PYTHON, "triton_capture.py", fixture, "dot-operand-load")
    issue = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.cp_async.issue"
    )
    read = next(
        operation for operation in snapshot["operations"]
        if operation["op"] == "triton.mem.read"
    )
    assert read["execution_evidence"]["method"] == "ttg-dot-operand-whole-tile-v1"
    assert read["lane_effects"]["lanes"] == issue["lane_effects"]["lanes"]
    assert read["lane_effects"]["regions"] == [read["region"]] * 128
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == verdict


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_shared_write_after_read_gets_membar(tmp_path: Path):
    reuse = """    %loaded = ttg.local_load %dst token %wait : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> -> tensor<64x16xf16, #blocked>
    %again = ttg.async_copy_global_to_local %src, %dst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    %again_group = ttg.async_commit_group tokens %again
    %again_wait = ttg.async_wait %again_group {num = 0 : i32}
    %reloaded = ttg.local_load %dst token %again_wait : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> -> tensor<64x16xf16, #blocked>
"""
    text = _TRANSPOSED_CP_ASYNC_TTGIR.replace("TRANSPOSED", "true")
    text = text.replace(reuse.splitlines(keepends=True)[0], reuse).replace(
        "#smem = #ttg.shared_memory",
        "#smem = #ttg.shared_memory\n"
        "#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>",
    ).replace(
        "-> tensor<64x16xf16, #blocked>",
        "-> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>",
    )
    fixture = tmp_path / "cp-async-reuse.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture(TRITON_PYTHON, "triton_capture.py", fixture, "cp-async-reuse")
    ops = [operation["op"] for operation in snapshot["operations"]]
    first_read = ops.index("triton.mem.read")
    second_issue = ops.index("triton.cp_async.issue", first_read)
    # Every thread reads the whole tile, so the reuse is ordered only by the
    # barrier Membar inserts before the second copy (WAR on the shared tile).
    assert "triton.barrier.arrive" in ops[first_read:second_issue]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_async_wait_gets_membar_after_unrelated_war_barrier(tmp_path: Path):
    # The WAR barrier before the reissue into %dst runs while the copy into
    # %next is still in flight, so only the barrier Membar places after the
    # final wait publishes %next to the other threads.
    pipelined = """    %loaded = ttg.local_load %dst token %wait : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> -> tensor<64x16xf16, #blocked>
    %next = ttg.local_alloc : () -> !ttg.memdesc<64x16xf16, #shared, #smem, mutable>
    %prefetch = ttg.async_copy_global_to_local %src, %next : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    %prefetch_group = ttg.async_commit_group tokens %prefetch
    %again = ttg.async_copy_global_to_local %src, %dst : tensor<64x16x!tt.ptr<f16>, #blocked> -> <64x16xf16, #shared, #smem, mutable>
    %again_group = ttg.async_commit_group tokens %again
    %again_wait = ttg.async_wait %again_group {num = 0 : i32}
    %prefetched = ttg.local_load %next token %again_wait : !ttg.memdesc<64x16xf16, #shared, #smem, mutable> -> tensor<64x16xf16, #blocked>
"""
    text = _TRANSPOSED_CP_ASYNC_TTGIR.replace("TRANSPOSED", "true")
    text = text.replace(pipelined.splitlines(keepends=True)[0], pipelined).replace(
        "#smem = #ttg.shared_memory",
        "#smem = #ttg.shared_memory\n"
        "#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>",
    ).replace(
        "-> tensor<64x16xf16, #blocked>",
        "-> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>",
    )
    fixture = tmp_path / "cp-async-pipelined.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture(TRITON_PYTHON, "triton_capture.py", fixture, "cp-async-pipelined")
    ops = [operation["op"] for operation in snapshot["operations"]]
    waits = [position for position, op in enumerate(ops) if op == "triton.cp_async.wait_group"]
    assert len(waits) == 2
    assert all(ops[position + 1] == "triton.barrier.arrive" for position in waits)
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_initialized_local_alloc_in_loop_gets_war_membar(tmp_path: Path):
    staged = """    %tile = tt.load %src : tensor<64x16x!tt.ptr<f16>, #blocked>
    %c0 = arith.constant 0 : i32
    %c1 = arith.constant 1 : i32
    %c2 = arith.constant 2 : i32
    scf.for %i = %c0 to %c2 step %c1 : i32 {
      %staged = ttg.local_alloc %tile : (tensor<64x16xf16, #blocked>) -> !ttg.memdesc<64x16xf16, #shared, #smem>
      %loaded = ttg.local_load %staged : !ttg.memdesc<64x16xf16, #shared, #smem> -> tensor<64x16xf16, #ttg.dot_op<{opIdx = 1, parent = #mma, kWidth = 2}>>
    }
"""
    lines = _TRANSPOSED_CP_ASYNC_TTGIR.replace("TRANSPOSED", "true").splitlines(keepends=True)
    start = next(position for position, line in enumerate(lines) if "%dst = ttg.local_alloc" in line)
    end = next(position for position, line in enumerate(lines) if "%loaded = ttg.local_load" in line)
    text = "".join(lines[:start] + [staged] + lines[end + 1:]).replace(
        "#smem = #ttg.shared_memory",
        "#smem = #ttg.shared_memory\n"
        "#mma = #ttg.nvidia_mma<{versionMajor = 2, versionMinor = 0, warpsPerCTA = [2, 2], instrShape = [16, 8]}>",
    )
    fixture = tmp_path / "staged-local-alloc.ttgir.mlir"
    fixture.write_text(text)
    _output, snapshot = _capture(TRITON_PYTHON, "triton_capture.py", fixture, "staged-local-alloc")
    ops = [operation["op"] for operation in snapshot["operations"]]
    shared_writes = [
        position for position, operation in enumerate(snapshot["operations"])
        if operation["op"] == "triton.mem.write" and ":shared:" in operation["region"]
    ]
    assert len(shared_writes) == 2
    # The second iteration overwrites the tile every thread read in the
    # first, which Membar orders with a barrier (WAR across the back edge).
    assert ops[shared_writes[1] - 2:shared_writes[1]] == [
        "triton.barrier.arrive",
        "triton.barrier.observe",
    ]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_cp_async_current_profile_rejects_incomplete_or_aggregate_evidence():
    _output, captured = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_cp_async.ttgir.mlir",
        "cp-async-evidence-rejection",
    )
    missing_lanes = json.loads(json.dumps(captured))
    issue = next(
        operation for operation in missing_lanes["operations"]
        if operation["op"] == "triton.cp_async.issue"
    )
    issue.pop("lane_effects")
    with pytest.raises(FrontendError) as caught:
        triton_snapshot_to_module(missing_lanes)
    assert caught.value.code == "TRITON_CP_ASYNC_LANE_TRANSFER_UNRESOLVED"

    aggregate = json.loads(json.dumps(captured))
    commit = next(
        operation for operation in aggregate["operations"]
        if operation["op"] == "triton.cp_async.commit_group"
    )
    commit["execution_evidence"].update({
        "transition": "ONCE",
        "rendezvous": "ALL_PROGRAM_INSTANCES",
    })
    with pytest.raises(FrontendError) as caught:
        triton_snapshot_to_module(aggregate)
    assert caught.value.code == "TRITON_EXECUTION_SCOPE_UNRESOLVED"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_cp_async_wait_is_thread_local_and_barrier_publishes():
    _output, captured = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        "triton_cp_async.ttgir.mlir",
        "cp-async-thread-local-control",
    )
    snapshot = json.loads(json.dumps(captured))
    snapshot["target"]["cta_threads"] = 2
    snapshot["actors"][0]["participants"] = [0, 1]
    for resource in snapshot["resources"]:
        if resource["kind"] == "cta_barrier":
            resource["capacity"] = 2

    issue = next(op for op in snapshot["operations"] if op["op"] == "triton.cp_async.issue")
    issue["lane_effects"] = {
        "lanes": [0],
        "transfer_offsets": [0, 1],
        "source_regions": [issue["lane_effects"]["source_regions"][0]],
        "destination_regions": [issue["lane_effects"]["destination_regions"][0]],
        "bytes": [4],
    }
    issue["execution_evidence"]["program_instances"] = [0]
    issue["execution_evidence"]["executors"] = [0]
    for operation in snapshot["operations"]:
        evidence = operation.get("execution_evidence")
        if operation["op"] in {
            "triton.cp_async.commit_group",
            "triton.cp_async.wait_group",
            "triton.barrier.arrive",
            "triton.barrier.observe",
        }:
            evidence["program_instances"] = [0, 1]
            evidence["executors"] = [0, 1]
        if operation["op"] == "triton.barrier.arrive":
            operation["count"] = 2
    read = next(op for op in snapshot["operations"] if op["op"] == "triton.mem.read")
    read["lane_effects"] = {
        "lanes": [1],
        "region_offsets": [0, 1],
        "regions": [issue["lane_effects"]["destination_regions"][0]],
    }

    safe_module = triton_snapshot_to_module(snapshot)
    safe_direct = check_module(safe_module, engine="maximal")
    safe_reference = check_module(safe_module.expand_reference(), engine="maximal")
    assert safe_direct["verdict"] == safe_reference["verdict"] == "SAFE"

    snapshot["operations"] = [
        operation for operation in snapshot["operations"]
        if not operation["op"].startswith("triton.barrier.")
    ]
    snapshot["resources"] = [
        resource for resource in snapshot["resources"]
        if resource["kind"] != "cta_barrier"
    ]
    snapshot["extraction"]["events_emitted"] = len(snapshot["operations"])
    unsafe_module = triton_snapshot_to_module(snapshot)
    unsafe_direct = check_module(unsafe_module, engine="maximal")
    unsafe_reference = check_module(unsafe_module.expand_reference(), engine="maximal")
    assert unsafe_direct["verdict"] == unsafe_reference["verdict"] == "DATA_HAZARD"
    assert unsafe_direct["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


def test_tilelang_assume_attribute_is_a_transparent_optimizer_fact():
    tools_root = str(ROOT / "tools")
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    from tilelang_capture import LaneWalker

    capture = types.SimpleNamespace(
        tvm=None,
        tirx=None,
        actor_ids=["lane:0"],
        scalar_bindings={},
    )
    body = type("SeqStmt", (), {"seq": []})()
    statement = type("AttrStmt", (), {
        "attr_key": "tl.assume",
        "body": body,
    })()

    LaneWalker(capture, 0).walk(statement, ("root",))


def test_tilelang_launch_grid_binds_a_direct_symbolic_block_extent():
    tools_root = str(ROOT / "tools")
    if tools_root not in sys.path:
        sys.path.insert(0, tools_root)
    from tilelang_capture import LaneWalker, TileLangCapture

    var_type = type("Var", (), {})
    num_tokens = var_type()
    capture = object.__new__(TileLangCapture)
    capture.func = types.SimpleNamespace(attrs={
        "thread_extent": {
            "blockIdx.x": num_tokens,
            "threadIdx.x": 64,
            "threadIdx.y": 1,
            "threadIdx.z": 1,
        },
    })
    capture.launch_grid = (8192, 1, 1)
    capture.launch_grid_explicit = True
    capture.tirx = types.SimpleNamespace(Var=var_type)
    capture.scalar_bindings = {}

    assert capture._thread_extents()["blockIdx.x"] == 8192
    assert capture.scalar_bindings == {num_tokens: 8192}

    capture.tvm = None
    capture.actor_ids = ["lane:0"]
    assert LaneWalker(capture, 0).bindings == {num_tokens: 8192}


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_ldmatrix_uses_hardware_footprint_of_wide_operands():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_ldmatrix_wide_device.tvm.json",
        "live-tilelang-ldmatrix-wide",
    )
    rows = [
        (lane, region["offset"], region["extent"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.read")
        if region["address_space"] == "SHARED"
    ]
    assert sorted(rows) == [(lane, lane * 16, 16) for lane in range(32)]
    fragments = {
        (region["offset"], region["extent"])
        for _lane, region in _lane_regions(snapshot, "tilelang.mem.write")
        if region["address_space"] == "REGISTER"
    }
    assert (0, 16) in fragments
    assert all(offset + extent <= 16 for offset, extent in fragments)
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_named_barrier_participants_are_the_lanes_reaching_it():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_named_barrier_device.tvm.json",
        "live-tilelang-named-barrier",
    )
    barriers = [resource for resource in snapshot["resources"] if resource["kind"] == "named_barrier"]
    assert len(barriers) == 1
    assert barriers[0]["capacity"] == 64
    arriving = sorted(
        lane
        for operation in snapshot["operations"]
        if operation["op"] == "tilelang.barrier.arrive"
        for lane in operation["instances"]
    )
    assert arriving == list(range(32, 96))
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_register_mma_after_grid_dependency_wait_is_register_only():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_register_mma_device.tvm.json",
        "live-tilelang-register-mma",
    )
    writes = sorted(
        (lane, region["offset"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.write")
        if region["address_space"] == "GLOBAL"
    )
    assert writes == [(lane, lane * 16 + 4 * i) for lane in range(32) for i in range(4)]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_wait_group_above_pending_groups_does_not_wait():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_deep_wait_group_device.tvm.json",
        "live-tilelang-deep-wait-group",
    )
    waits = [
        operation for operation in snapshot["operations"]
        if operation["op"] == "tilelang.cp_async.wait_group"
    ]
    assert waits
    assert {
        count for operation in waits for count in _compact_field_values(operation, "wait_group_count")
    } == {8}
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] != "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_loaded_scalar_binding_stays_opaque_for_values():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_opaque_scalar_device.tvm.json",
        "live-tilelang-opaque-scalar",
    )
    reads = sorted(
        (lane, region["offset"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.read")
        if region["address_space"] == "GLOBAL"
    )
    assert reads == [(lane, lane * 4) for lane in range(128)]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_loaded_scalar_used_as_address_is_unresolved():
    process = _run_capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_opaque_address_device.tvm.json",
        "live-tilelang-opaque-address",
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] in {
        "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED",
        "TILELANG_STATIC_VALUE_UNRESOLVED",
    }


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_cp_async_mixed_arrivals_do_not_invent_barrier_generations():
    process = _run_capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_cp_async_mixed_arrivals_device.tvm.json",
        "live-tilelang-cp-async-mixed-arrivals",
    )
    assert process.returncode == 2
    diagnostic = json.loads(process.stdout)["diagnostics"][0]
    assert diagnostic["code"] == "TILELANG_MBARRIER_PHASE_UNRESOLVED"
    assert diagnostic["details"]["arrival_kinds"] == ["cp_async_noinc", "plain"]


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_cp_async_noinc_arrival_publishes_completed_copies():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_cp_async_noinc_device.tvm.json",
        "live-tilelang-cp-async-noinc",
    )
    arrivals = sorted(
        lane
        for operation in snapshot["operations"]
        if operation["op"] == "tilelang.cp_async.mbarrier_arrive"
        for lane in operation["instances"]
    )
    assert arrivals == list(range(128))
    waits = [
        operation for operation in snapshot["operations"]
        if operation["op"] == "tilelang.cp_async.wait_group"
    ]
    assert not waits
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_sync_warp_is_a_per_warp_barrier():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_sync_warp_device.tvm.json",
        "live-tilelang-sync-warp",
    )
    warp_barriers = sorted(
        resource["participants"]
        for resource in snapshot["resources"]
        if resource["kind"] == "named_barrier" and resource["capacity"] == 32
    )
    assert len(warp_barriers) == 2
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize("fixture", ["tilelang_warp_allreduce_device.tvm.json",
                                     "tilelang_warp_allreduce_unused_barrier_device.tvm.json"])
def test_tilelang_lowered_warp_allreduce_is_register_only(fixture):
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        fixture,
        "live-tilelang-warp-allreduce",
    )
    writes = sorted(
        (lane, region["offset"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.write")
        if region["address_space"] == "GLOBAL"
    )
    assert writes == [(lane, lane * 4) for lane in range(32)]
    assert not any(resource["kind"] == "named_barrier" for resource in snapshot["resources"])
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


def test_wgmma_layout_hull_falls_back_to_ptx_roles():
    # MN-major 64B swizzle, one MN atom: LBO (4 KiB) strides MN atoms and
    # SBO (512 B) strides 8-row K groups, so a K16 slice reads 1 KiB.
    arguments = {
        "mn": 32, "k": 16, "element_bytes": 2, "k_major": False,
        "layout_type": 2, "leading": 256, "stride": 32, "start": 28672,
    }
    assert _wgmma_layout_hull(**arguments, hedge_swapped=False) == (28672, 29696)
    assert _wgmma_layout_hull(**arguments)[1] > 32768


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_concurrent_atomics_on_one_counter_are_safe():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_atomic_add_device.tvm.json",
        "live-tilelang-atomic-add",
    )
    atomics = sorted(
        (lane, region["offset"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.atomic")
    )
    assert atomics == sorted(
        [(lane, lane % 4 * 4) for lane in range(128)]
        + [(lane, 0) for lane in range(128)]
        + [(lane, lane * 4) for lane in range(4)]
    )
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_plain_read_of_an_atomic_counter_races():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_atomic_race_device.tvm.json",
        "live-tilelang-atomic-race",
    )
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_vector_atomic_adds_four_elements_from_registers():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_vector_atomic_device.tvm.json",
        "live-tilelang-vector-atomic",
    )
    atomics = sorted(
        (lane, region["offset"], region["extent"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.atomic")
    )
    assert atomics == [(lane, lane % 4 * 16, 16) for lane in range(128)]
    register_reads = [
        region["extent"]
        for _lane, region in _lane_regions(snapshot, "tilelang.mem.read")
        if region["address_space"] != "GLOBAL"
    ]
    assert register_reads == [16] * 128
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_vectorized_atomic_reads_its_shared_value_vector():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_vectorized_atomic_device.tvm.json",
        "live-tilelang-vectorized-atomic",
    )
    atomics = sorted(
        (region["offset"], region["extent"])
        for _lane, region in _lane_regions(snapshot, "tilelang.mem.atomic")
    )
    assert atomics == [(offset, 16) for offset in range(0, 1024, 16)]
    shared_reads = sorted(
        (region["offset"], region["extent"])
        for _lane, region in _lane_regions(snapshot, "tilelang.mem.read")
        if region["address_space"] == "SHARED"
    )
    assert sum(extent for _offset, extent in shared_reads) == 1024
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_data_dependent_register_branch_reads_both_arms():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_register_branch_device.tvm.json",
        "live-tilelang-register-branch",
    )
    reads = sorted(
        (lane, region["offset"])
        for lane, region in _lane_regions(snapshot, "tilelang.mem.read")
        if region["address_space"] == "GLOBAL"
    )
    # A[tx] for the initial value, A[127 - tx] in the condition and the arm.
    assert reads == sorted(
        [(lane, lane * 4) for lane in range(128)]
        + [(lane, (127 - lane) * 4) for lane in range(128)] * 2
    )
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_warp_votes_are_register_exchanges():
    _output, snapshot = _capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_warp_vote_device.tvm.json",
        "live-tilelang-warp-vote",
    )
    assert sorted({operation["op"] for operation in snapshot["operations"]}) == [
        "tilelang.mem.read",
        "tilelang.mem.write",
    ]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_data_dependent_shared_branch_is_unsupported():
    process = _run_capture(
        TILELANG_PYTHON,
        "tilelang_capture.py",
        "tilelang_shared_branch_device.tvm.json",
        "live-tilelang-shared-branch",
    )
    assert process.returncode == 2
    assert json.loads(process.stdout)["diagnostics"][0]["code"] == "TILELANG_DYNAMIC_CONTROL_FLOW_UNSUPPORTED"


def _atomic_ttgir(tmp_path: Path, scope: str, read_back: bool, printed: bool = False) -> Path:
    path = tmp_path / f"atomic-{scope}-{read_back}-{printed}.ttgir.mlir"
    read = (
        "\n    %loaded = tt.load %pointers : tensor<128x!tt.ptr<f32>, #blocked>"
        if read_back else ""
    ) + (
        "\n    tt.grid_dependency_wait"
        '\n    tt.print " old: " {hex = false, isSigned = array<i32: 0>} : %old : tensor<128xf32, #blocked>'
        "\n    tt.grid_dependency_launch_dependents"
        if printed else ""
    )
    path.write_text(f"""\
#blocked = #ttg.blocked<{{sizePerThread = [1], threadsPerWarp = [32], warpsPerCTA = [4], order = [0]}}>
module attributes {{"ttg.num-ctas" = 1 : i32, "ttg.num-warps" = 4 : i32, ttg.target = "cuda:90", "ttg.threads-per-warp" = 32 : i32}} {{
  tt.func public @atomic_counter(%base: !tt.ptr<f32>) {{
    %ones = arith.constant dense<1.000000e+00> : tensor<128xf32, #blocked>
    %mask = arith.constant dense<true> : tensor<128xi1, #blocked>
    %pointers = tt.splat %base : !tt.ptr<f32> -> tensor<128x!tt.ptr<f32>, #blocked>
    %old = tt.atomic_rmw fadd, relaxed, {scope}, %pointers, %ones, %mask : (tensor<128x!tt.ptr<f32>, #blocked>, tensor<128xf32, #blocked>, tensor<128xi1, #blocked>) -> tensor<128xf32, #blocked>{read}
    tt.return
  }}
}}
""")
    return path


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_atomic_rmw_on_one_counter_is_safe_but_races_a_plain_read(tmp_path: Path):
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        _atomic_ttgir(tmp_path, "gpu", read_back=False),
        "atomic-counter",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == ["triton.mem.atomic"]
    assert snapshot["operations"][0]["lane_effects"]["lanes"] == list(range(128))
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"

    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        _atomic_ttgir(tmp_path, "gpu", read_back=True),
        "atomic-counter-read",
    )
    result = check_module(triton_snapshot_to_module(snapshot), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_print_and_grid_dependency_control_have_no_footprint(tmp_path: Path):
    _output, snapshot = _capture(
        TRITON_PYTHON,
        "triton_capture.py",
        _atomic_ttgir(tmp_path, "gpu", read_back=False, printed=True),
        "atomic-counter-print",
    )
    assert [operation["op"] for operation in snapshot["operations"]] == ["triton.mem.atomic"]
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_cta_scope_atomic_is_safe_within_selected_cta(tmp_path: Path):
    process = _run_capture(
        TRITON_PYTHON,
        "triton_capture.py",
        _atomic_ttgir(tmp_path, "cta", read_back=False),
        "atomic-counter-cta",
    )
    assert process.returncode == 0, process.stdout
    assert check_module(triton_snapshot_to_module(json.loads(process.stdout)), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_proxy_fence_covers_shared_memory_beyond_64k():
    # TileLang 2c31e071 test_gemm_sliced_shared_operand, 128 KiB shared arena.
    # B starts at 65536, beyond the former fixed fence window.
    _output, snapshot = _capture(
        TILELANG_PYTHON, "tilelang_capture.py",
        "tilelang_sliced_gemm_128k_device.tvm.json", "sliced-gemm-128k",
        launch_grid=(2, 2, 1),
    )
    assert snapshot["target"]["shared_memory_bytes"] >= 131072
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"
    # Reproduce the old truncated visibility: the same IR must still report
    # the uncovered B writes, rather than suppressing WGMMA conflicts.
    for operation in snapshot["operations"]:
        if operation["op"] == "tilelang.proxy_fence":
            operation["fields"]["regions"] = ["region:shared:buf_dyn_shmem:0:65536"]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "DATA_HAZARD"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_tensormap_ir_derives_4d_boxes_and_split_store(tmp_path: Path):
    _output, snapshot = _capture(
        TRITON_PYTHON, "triton_capture.py", "triton_tma_4d_derived.ttgir.mlir", "tma-4d-derived",
        binding_evidence="triton_tma_4d_derived.binding.json",
    )
    assert check_module(triton_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"
    assert snapshot["operations"][-1]["wait_mode"] == "GLOBAL"
    # Nonzero outer coordinate checks that TTGIR coordinates are outermost
    # first, even though tensormap_create dimensions are innermost first.
    fixture = tmp_path / "offset.mlir"
    fixture.write_text((FIXTURES / "triton_tma_4d_derived.ttgir.mlir").read_text().replace(
        "%in_desc_0[%c0_i32, %c0_i32, %c0_i32, %c0_i32]",
        "%in_desc_0[%c1_i32, %c0_i32, %c0_i32, %c0_i32]",
    ))
    evidence = json.loads((FIXTURES / "triton_tma_4d_derived.binding.json").read_text())
    evidence["source_ir_sha256"] = hashlib.sha256(fixture.read_bytes()).hexdigest()
    binding = tmp_path / "offset.binding.json"
    binding.write_text(json.dumps(evidence))
    _output, shifted = _capture(TRITON_PYTHON, "triton_capture.py", fixture, "offset-tma", binding_evidence=binding)
    # Includes the descriptor's base region; check the actual issue sources.
    regions = {r["id"]: r for r in shifted["regions"]}
    sources = [regions[t["source_region"]] for o in shifted["operations"] if o["op"] == "triton.tma_load.issue" for t in o["transfers"]]
    assert min(r["offset"] for r in sources if r["allocation_id"] == "global:In") == 1024


def test_triton_slice_blocked_replication_keeps_canonical_store_owners():
    from triton_capture import TritonCapture
    layout = ('tensor<32x!tt.ptr<f32>, #ttg.slice<{dim = 0, parent = '
              '#ttg.blocked<{sizePerThread = [2, 2], threadsPerWarp = [2, 16], '
              'warpsPerCTA = [4, 1], order = [1, 0]}>}>>')
    owners = TritonCapture._blocked_lane_elements(layout, [32], list(range(128)), representative_only=True)
    assert owners == {lane: (2 * lane, 2 * lane + 1) for lane in range(16)}
    reads = TritonCapture._blocked_lane_elements(layout, [32], list(range(128)), representative_only=False)
    assert len(reads) == 128
    assert all(reads[lane] == reads[lane % 16] for lane in range(128))


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize("count", [1, 2, 4])
def test_tilelang_stmatrix_rows_follow_matrix_count(count):
    _, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py",
                           f"tilelang_stmatrix_x{count}_device.tvm.json", f"stmatrix-x{count}")
    templates = {op["id"] for op in snapshot["operations"]
                 if op["evidence"]["ir_operation"] == "tl.ptx_stmatrix"}
    writes = [op for op in snapshot["operations"] if op["id"] in templates]
    assert len(writes) == 1
    assert writes[0]["instances"] == list(range(8 * count))
    # Each hardware row spans 16 bytes, independently of the per-lane
    # register fragment extent attached to access_ptr by TileLang.
    rows = [(lane, region["offset"], region["extent"])
            for lane, region in _lane_regions(snapshot, "tilelang.mem.write")
            if region["address_space"] == "SHARED" and region["extent"] == 16]
    assert sorted(rows) == [(lane, lane * 16, 16) for lane in range(8 * count)]
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_arrive_expect_covers_multiple_tma_panels():
    _, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py",
                           "tilelang_tma_multi_panel_device.tvm.json", "multi-panel-generation",
                           launch_grid=(4, 4, 1), descriptor_evidence="tilelang_tma_multi_panel_descriptors.json")
    issues = [op for op in snapshot["operations"] if op["op"] == "tilelang.tma_load.issue"]
    assert len(issues) == 20  # Four iterations, one A panel and four B panels.
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
@pytest.mark.parametrize("fixture,verdict", [
    ("tilelang_cta_vote_safe.tvm.json", "SAFE"),
    ("tilelang_cta_vote_divergent.tvm.json", "SYNC_HAZARD"),
    ("tilelang_explicit128_safe.tvm.json", "SAFE"),
    ("tilelang_explicit128_race.tvm.json", "DATA_HAZARD"),
    ("tilelang_multidim_safe.tvm.json", "SAFE"),
    ("tilelang_multidim_race.tvm.json", "DATA_HAZARD"),
    ("tilelang_batch_reduce_safe.tvm.json", "SAFE"),
    ("tilelang_batch_reduce_reuse_race.tvm.json", "DATA_HAZARD"),
])
def test_tilelang_multidimensional_threads_and_external_workspace(fixture, verdict):
    _, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py", fixture, "coverage-20261030")
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == verdict


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_static_ballot_mask_orders_only_reduction_participants():
    _, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py", "tilelang_partial_ballot_reduce_device.tvm.json", "partial-ballot-reduce")
    barriers = [r for r in snapshot['resources'] if r['kind'] == 'named_barrier' and r['capacity'] == 4]
    assert barriers
    assert all(r['participants'] == ['lane:0000', 'lane:0001', 'lane:0002', 'lane:0003'] for r in barriers)
    result = check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")
    assert result['verdict'] == 'DATA_HAZARD'
    assert result['witness']['allocation'] == 'C'  # final unguarded output store, not the ordered reduction scratch


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang capture environment unavailable")
def test_tilelang_rng_distributions_keep_all_output_writes():
    _, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py",
                           "tilelang_rng_distributions_device.tvm.json", "rng-distributions", launch_grid=(4, 1, 1))
    writes = [r for _, r in _lane_regions(snapshot, "tilelang.mem.write")]
    assert {r["allocation_id"] for r in writes} == {"A", "B", "C", "D", "E"}
    assert sum(r["extent"] for r in writes) == 128 * (4 + 4 + 8 + 4 + 8)
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang environment unavailable")
@pytest.mark.parametrize("mutable", [False, True])
def test_recorded_tensor_values_require_readonly_input(tmp_path, mutable):
    fixture = FIXTURES / ("tilelang_constant_input_" + ("mutable" if mutable else "readonly") + ".tvm.json")
    evidence = tmp_path / "binding.json"
    evidence.write_text(json.dumps({"evidence_version": "tilelang-binding-evidence-v1",
        "source_ir_sha256": hashlib.sha256(fixture.read_bytes()).hexdigest(),
        "scalar_values": {}, "buffer_values": {"I": [16, 32]},
        "pointer_table_allocations": {}, "provenance": {"kind": "test"}}))
    process = _run_capture(TILELANG_PYTHON, "tilelang_capture.py", fixture, "constant-input", binding_evidence=evidence)
    snapshot = json.loads(process.stdout)
    if mutable:
        assert process.returncode != 0
        assert snapshot["diagnostics"][0]["code"] == "TILELANG_BINDING_EVIDENCE_INVALID"
    else:
        process.check_returncode()
        assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"
        assert len(_lane_regions(snapshot, "tilelang.mem.write")) == 16


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang environment unavailable")
@pytest.mark.parametrize("fixture", ["tilelang_empty_cuda.tvm.json", "tilelang_true_device_assert.tvm.json"])
def test_official_cuda_noop_and_true_device_assert(fixture):
    _, snapshot = _capture(TILELANG_PYTHON, "tilelang_capture.py", fixture, "official-cuda")
    assert not _lane_regions(snapshot, "tilelang.mem.write")
    assert check_module(tilelang_snapshot_to_module(snapshot), engine="maximal")["verdict"] == "SAFE"


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang environment unavailable")
def test_false_device_assert_is_not_erased(tmp_path):
    raw = json.loads((FIXTURES / 'tilelang_true_device_assert.tvm.json').read_text())
    calls = [node for node in raw['nodes'] if node.get('type') == 'tirx.Call'
             and raw['nodes'][node['data']['op']].get('data') == 'tl.device_assert_with_msg']
    assert len(calls) == 1
    arguments = raw['nodes'][calls[0]['data']['args']]['data']
    predicate = raw['nodes'][arguments[0]]
    assert predicate['type'] == 'ir.IntImm' and predicate['data']['dtype'] == 'bool'
    assert predicate['data']['value'] == 1
    predicate['data']['value'] = 0
    fixture = tmp_path / 'false-assert.tvm.json'
    fixture.write_text(json.dumps(raw))
    result = _run_capture(TILELANG_PYTHON, 'tilelang_capture.py', fixture, 'false-assert')
    assert result.returncode != 0
    assert json.loads(result.stdout)['diagnostics'][0]['code'] == 'TILELANG_ASSERTION_UNRESOLVED'


@pytest.mark.skipif(not TILELANG_PYTHON.exists(), reason="pinned TileLang environment unavailable")
@pytest.mark.parametrize("extent,verdict", [("negative", "SAFE"), ("positive", "DATA_HAZARD")])
def test_signed_cuda_loop_upper_bound(extent, verdict):
    _, snapshot = _capture(TILELANG_PYTHON, 'tilelang_capture.py', 'tilelang_signed_loop_' + extent + '.tvm.json', 'signed-loop')
    assert check_module(tilelang_snapshot_to_module(snapshot), engine='maximal')['verdict'] == verdict
    if extent == 'negative':
        assert not _lane_regions(snapshot, 'tilelang.mem.write')


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize("elements, verdict", [(16, "DATA_HAZARD"), (128, "SAFE")])
@pytest.mark.parametrize("explicit_store", [False, True])
def test_triton_shared_stores_keep_physical_replicas(tmp_path, elements, verdict, explicit_store):
    text = (FIXTURES / "triton_initialized_local.ttgir.mlir").read_text().replace("16xi8", f"{elements}xi8")
    if explicit_store:
        line = next(line for line in text.splitlines() if "%dst = ttg.local_alloc" in line)
        ty = f"!ttg.memdesc<{elements}xi8, #shared, #smem, mutable>"
        text = text.replace(line, f"    %dst = ttg.local_alloc : () -> {ty}\n    ttg.local_store %payload, %dst : tensor<{elements}xi8, #blocked> -> {ty}")
    path = tmp_path / "physical-writers.ttgir.mlir"
    path.write_text(text)
    _, snapshot = _capture(TRITON_PYTHON, "triton_capture.py", path, "physical-shared-writers")
    store = next(op for op in snapshot["operations"] if op["op"] == "triton.mem.write")
    assert store["lane_effects"]["lanes"] == list(range(128))
    assert store["execution_evidence"]["redundant_policy"] == "STORE_ALL"
    result = check_module(triton_snapshot_to_module(snapshot))
    assert result["verdict"] == verdict


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_final_ptx_late_accumulator_and_replicated_writes_are_both_reported(tmp_path):
    _, snapshot = _capture(
        TRITON_PYTHON, "triton_capture.py", "triton_wgmma_replicated_late_accumulator.ttgir.mlir",
        "physical-wgmma-regression", final_ptx="triton_wgmma_replicated_late_accumulator.ptx",
    )
    module = triton_snapshot_to_module(snapshot)
    result = check_module(module, check_ptx_registers=True)
    assert result["verdict"] == "DATA_HAZARD"
    regions = {region["id"]: region for region in snapshot["regions"]}
    initial_writes = [op for op in snapshot["operations"] if op["op"] == "triton.mem.write"][:2]
    # Independently enumerated PTX: A has 256 x 16B stores; B has 1024 x 16B.
    assert [sum(regions[r]["extent"] for r in op["lane_effects"]["regions"])
            for op in initial_writes] == [4096, 16384]
    gaps = [d for d in result["diagnostics"] if d["code"] == "WGMMA_REGISTER_FENCE_MISSING"]
    assert [d["details"]["mma_line"] for d in gaps] == [122, 142]
    # Repair only the register ordering in an isolated test input. The shared
    # WAW must remain observable rather than being hidden by the SYNC finding.
    text = (FIXTURES / "triton_wgmma_replicated_late_accumulator.ptx").read_text()
    lines = text.splitlines()
    for line in [142, 122]:
        lines.insert(line - 1, "wgmma.fence.sync.aligned;")
    result = check_module(module, final_ptx="\n".join(lines), check_ptx_registers=True)
    assert result["verdict"] == "DATA_HAZARD"
    assert not any(d["code"] == "WGMMA_REGISTER_FENCE_MISSING" for d in result["diagnostics"])
    snapshot["profile"]["final_ptx_evidence"]["text"] += "\n"
    with pytest.raises(FrontendError, match="digest/size"):
        triton_snapshot_to_module(snapshot)


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_duplicate_arrive_expect_reports_current_phase_underflow(tmp_path):
    text = (FIXTURES / "triton_tma_load.ttgir.mlir").read_text()
    expect = next(line for line in text.splitlines() if "ttng.barrier_expect" in line)
    text = text.replace(expect, expect + "\n" + expect)
    path = tmp_path / "duplicate-arrive.ttgir.mlir"
    path.write_text(text)
    _, snapshot = _capture_with_tma_layout(tmp_path, path, "duplicate-arrive", "global:tma")
    arrivals = [op for op in snapshot["operations"] if op["op"] == "triton.mbarrier.arrive_expect_tx"]
    assert [op["generation"] for op in arrivals] == [0, 0]
    result = check_module(triton_snapshot_to_module(snapshot))
    assert result["verdict"] == "SYNC_HAZARD"
    assert result["diagnostics"][0]["code"] == "ARRIVAL_UNDERFLOW"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize("two_arrivals", [False, True])
def test_triton_tma_panels_and_partial_arrivals_share_generation(tmp_path, two_arrivals):
    text = (FIXTURES / "triton_tma_load.ttgir.mlir").read_text()
    alloc = next(line for line in text.splitlines() if "%destination = ttg.local_alloc" in line)
    expect = next(line for line in text.splitlines() if "ttng.barrier_expect" in line)
    load = next(line for line in text.splitlines() if "ttng.async_tma_copy_global_to_local" in line)
    text = text.replace(alloc, alloc + "\n" + alloc.replace("%destination", "%other"))
    if two_arrivals:
        text = text.replace("ttng.init_barrier %barrier_arg, 1", "ttng.init_barrier %barrier_arg, 2")
        text = text.replace(load, load + "\n" + expect + "\n" + load.replace("%destination", "%other"))
    else:
        text = text.replace(expect, expect.replace("16384", "32768"))
        text = text.replace(load, load + "\n" + load.replace("%destination", "%other"))
    path = tmp_path / "panels.ttgir.mlir"
    path.write_text(text)
    _, snapshot = _capture_with_tma_layout(tmp_path, path, "panels", "global:tma")
    issues = [op for op in snapshot["operations"] if op["op"] == "triton.tma_load.issue"]
    assert [op["generation"] for op in issues] == [0, 0]
    assert check_module(triton_snapshot_to_module(snapshot))["verdict"] == "SAFE"


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
def test_triton_false_tma_arrival_and_issue_do_not_start_generation(tmp_path):
    text = (FIXTURES / "triton_tma_load.ttgir.mlir").read_text().replace("arith.constant true", "arith.constant false")
    text = '\n'.join(line for line in text.splitlines() if 'ttng.wait_barrier' not in line and '%loaded = ttg.local_load' not in line)
    path = tmp_path / 'false-tma.ttgir.mlir'
    path.write_text(text)
    _, snapshot = _capture(TRITON_PYTHON, 'triton_capture.py', path, 'false-tma')
    assert not any(op['op'] == 'triton.tma_load.issue' for op in snapshot['operations'])
    assert check_module(triton_snapshot_to_module(snapshot))['verdict'] == 'SAFE'


def test_legacy_dialect_names_preserve_asm_locations_and_comments():
    from triton_capture import _canonical_dialect_names
    text = ('// triton_gpu.local_store\n'
            '#x = #triton_gpu.blocked<{order = [0]}>\n'
            'module attributes {"triton_gpu.num-warps" = 4 : i32} {\n'
            'tt.elementwise_inline_asm "triton_gpu.local_store", "=r", pure = true\n'
            '} loc("triton_gpu.num-warps")\n')
    actual = _canonical_dialect_names(text)
    assert '#ttg.blocked<' in actual
    assert '"ttg.num-warps" = 4' in actual
    assert '// triton_gpu.local_store' in actual
    assert 'inline_asm "triton_gpu.local_store"' in actual
    assert 'loc("triton_gpu.num-warps")' in actual


@pytest.mark.skipif(not TRITON_PYTHON.exists(), reason="pinned Triton capture environment unavailable")
@pytest.mark.parametrize('fixture', ['triton_cp_async.ttgir.mlir', 'triton_static_disjoint_memory.ttgir.mlir'])
def test_legacy_dialect_namespace_has_same_capture_semantics(tmp_path, fixture):
    current = FIXTURES / fixture
    legacy = tmp_path / fixture
    legacy.write_text(current.read_text().replace('ttg.', 'triton_gpu.').replace('ttng.', 'triton_nvidia_gpu.'))
    _, expected = _capture(TRITON_PYTHON, 'triton_capture.py', current, 'dialect-parity')
    _, actual = _capture(TRITON_PYTHON, 'triton_capture.py', legacy, 'dialect-parity')
    assert actual['operations'] == expected['operations']
    assert actual['target'] == expected['target']
    assert actual['regions'] == expected['regions']
