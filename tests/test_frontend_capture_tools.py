from __future__ import annotations

import hashlib
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
    grid_ctas: int | None = 1,
    launch_grid: tuple[int, ...] | None = None,
    selected_cta: tuple[int, int, int] = (0, 0, 0),
    descriptor_evidence: str | None = None,
    binding_evidence: str | Path | None = None,
):
    environment = dict(os.environ)
    environment["PYTHONHASHSEED"] = str(seed)
    fixture_path = Path(fixture)
    if not fixture_path.is_absolute():
        fixture_path = FIXTURES / fixture_path
    command = [str(python), str(ROOT / "tools" / tool), str(fixture_path), "--source", source]
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
    grid_ctas: int | None = 1,
    launch_grid: tuple[int, ...] | None = None,
    selected_cta: tuple[int, int, int] = (0, 0, 0),
    descriptor_evidence: str | None = None,
    binding_evidence: str | Path | None = None,
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
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    resources = [operation["resource"] for operation in snapshot["operations"]]
    assert len(set(resources[:3])) == 1
    assert len(set(resources[3:])) == 1
    assert resources[0] != resources[3]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"

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
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    assert json.loads(missing.stdout)["diagnostics"][0]["code"] == "TRITON_TMA_DESCRIPTOR_UNRESOLVED"

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
    module = module.replace("warpsPerCTA = [1, 4]", "warpsPerCTA = [1, 8]")
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
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    assert check_module(triton_snapshot_to_module(waited_snapshot), engine="crosscheck")["verdict"] == "SAFE"

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
    result = check_module(triton_snapshot_to_module(missing_snapshot), engine="crosscheck")
    assert missing_snapshot["operations"][-1]["region"].startswith("region:register:")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["diagnostics"][0]["code"] == "UNORDERED_CONFLICT"


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
    ttng.init_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>""",
    ).replace(
        "    %3 = ttng.warp_group_dot_wait %2 {pendings = 0 : i32} : tensor<128x64xf32, #mma>",
        "    ttng.arrive_barrier %bar, 1 : !ttg.memdesc<1xi64, #barrier, #smem, mutable>",
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
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    ]
    assert snapshot["extraction"]["static_expansions"] == 1
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
    result = check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")
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
        "triton.mem.read",
    ]
    issue = snapshot["operations"][0]
    assert issue["source_region"] == "region:global:global-arg:0:0:8"
    assert issue["destination_region"] == "region:shared:shared:ssa:6:0:0:8"
    assert issue["bytes"] == 8
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
        "triton.mem.read",
    ]
    assert snapshot["operations"][0]["region"] == snapshot["operations"][1]["region"]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
        "triton.mem.read",
    ]
    assert snapshot["operations"][0]["region"] == snapshot["operations"][1]["region"]
    assert check_module(triton_snapshot_to_module(snapshot), engine="crosscheck")["verdict"] == "SAFE"


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
  tt.func public @async_loop_carried(%base: !tt.ptr<i8>) {
    %zero = arith.constant 0 : index
    %one = arith.constant 1 : index
    %range = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32, #blocked>
    %base_vector = tt.splat %base : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>, #blocked>
    %src = tt.addptr %base_vector, %range : tensor<16x!tt.ptr<i8>, #blocked>, tensor<16xi32, #blocked>
    %dst = ttg.local_alloc : () -> !ttg.memdesc<16xi8, #shared, #smem, mutable>
    %copy = ttg.async_copy_global_to_local %src, %dst : tensor<16x!tt.ptr<i8>, #blocked> -> <16xi8, #shared, #smem, mutable>
    %group = ttg.async_commit_group tokens %copy
    %carried = scf.for %i = %zero to %one step %one iter_args(%token = %group) -> (!ttg.async.token) {
      scf.yield %token : !ttg.async.token
    }
    %wait = ttg.async_wait %carried {num = 0 : i32}
    %loaded = ttg.local_load %dst token %wait : !ttg.memdesc<16xi8, #shared, #smem, mutable> -> tensor<16xi8, #blocked>
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
        "triton.mem.read",
    ]


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
    result = check_module(triton_snapshot_to_module(snapshot))
    assert result["verdict"] == "SAFE"
    assert "selected CTA [0, 0, 0] from launch grid [2, 3, 1]" in result["manifest"]["scope"]


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
    assert snapshot["target"]["cta_threads"] == 256
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
