from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from pipeguard.checker import check_module
from pipeguard.frontends import (
    semantic_digest,
    semantic_signature,
    tilelang_snapshot_to_module,
    triton_snapshot_to_module,
)
from pipeguard.frontends.tilelang import _EVENT_TYPES as TILELANG_EVENT_TYPES
from pipeguard.frontends.tilelang import _event as tilelang_event
from pipeguard.frontends.common import FrontendError
from pipeguard.frontends.triton import _EVENT_TYPES as TRITON_EVENT_TYPES
from pipeguard.frontends.triton import _event as triton_event
from pipeguard.model import IMPLEMENTED_EVENTS

FIXTURES = Path(__file__).parent / "fixtures"
TILELANG = FIXTURES / "tilelang_safe_handoff.snapshot.json"
TRITON = FIXTURES / "triton_safe_handoff.snapshot.json"
TILELANG_ASYNC = FIXTURES / "tilelang_async_pipeline.snapshot.json"
TRITON_ASYNC = FIXTURES / "triton_async_pipeline.snapshot.json"


def _load(path: Path) -> dict:
    return json.loads(path.read_text())


def _as_v2(snapshot: dict, prefix: str, stage: str) -> dict:
    snapshot["snapshot_version"] = f"{prefix}-snapshot-v2"
    snapshot["producer"]["adapter_version"] = f"{prefix}-pgir-v2"
    snapshot["profile"] = {
        "id": "sm90-single-cta-static-v1",
        "source_ir_sha256": "0" * 64,
        "compiler_ir_stage": stage,
        "target_rule": f"{prefix}-fixture-target-v1",
    }
    operation_count = len(snapshot["operations"])
    snapshot["extraction"].update({
        "static_expansions": 0,
        "executable_nodes_seen": operation_count,
        "classified_nodes": operation_count,
        "emitting_nodes": operation_count,
        "elided_nodes": 0,
        "events_emitted": operation_count,
        "allocation_identity_evidence": "fixture-allocation-identity-v1",
        "alias_evidence": "fixture-exact-alias-v1",
        "participant_evidence": "fixture-static-participants-v1",
    })
    for index, operation in enumerate(snapshot["operations"]):
        operation["evidence"] = {
            "source_location": f"fixture:{index}",
            "activation_path": ["root", index],
            "recognizer_rule": f"fixture-{prefix}-rule-v1",
            "ir_operation": operation["op"],
        }
    return snapshot


def _typed_protocol_snapshot(path: Path, prefix: str, protocol: str) -> dict:
    snapshot = _load(path)
    snapshot["target"]["cta_threads"] = 128
    snapshot["actors"] = [{
        "id": "owner",
        "kind": "WARPGROUP",
        "participants": list(range(128)),
    }]
    if protocol == "wgmma":
        snapshot["resources"] = [{
            "id": "wgmma",
            "kind": "async_group",
            "participants": ["owner"],
            "capacity": 8,
            "protocol": "WGMMA",
            "scope": "CTA",
        }]
        snapshot["regions"] = [
            {"id": "a", "allocation_id": "a", "address_space": "SHARED", "offset": 0, "extent": 64},
            {"id": "b", "allocation_id": "b", "address_space": "SHARED", "offset": 0, "extent": 64},
            {"id": "acc", "allocation_id": "acc", "address_space": "REGISTER", "offset": 0, "extent": 64},
        ]
        operations = [
            {"id": "fence", "op": f"{prefix}.wgmma.fence_operand", "resource": "wgmma", "group_owner": "owner", "regions": ["a", "b"]},
            {"id": "issue", "op": f"{prefix}.wgmma.issue", "resource": "wgmma", "group_owner": "owner", "regions": ["a", "b"], "destination_region": "acc"},
            {"id": "commit", "op": f"{prefix}.wgmma.commit_group", "resource": "wgmma", "group_owner": "owner", "group_seq": 0},
            {"id": "operands", "op": f"{prefix}.wgmma.wait_group", "resource": "wgmma", "group_owner": "owner", "wait_group_count": 0, "completion_kind": "OPERANDS_RELEASED"},
            {"id": "result", "op": f"{prefix}.wgmma.wait_group", "resource": "wgmma", "group_owner": "owner", "wait_group_count": 0, "completion_kind": "RESULT_READY"},
            {"id": "read", "op": f"{prefix}.mem.read", "region": "acc"},
        ]
    else:
        snapshot["resources"] = [{
            "id": "stores",
            "kind": "async_group",
            "participants": ["owner"],
            "capacity": 8,
            "protocol": "TMA_STORE",
            "scope": "CTA",
        }]
        snapshot["regions"] = [
            {"id": "source", "allocation_id": "source", "address_space": "SHARED", "offset": 0, "extent": 64},
            {"id": "destination", "allocation_id": "destination", "address_space": "GLOBAL", "offset": 0, "extent": 64},
        ]
        operations = [
            {"id": "write", "op": f"{prefix}.mem.write", "region": "source"},
            {
                "id": "proxy", "op": f"{prefix}.proxy_fence", "from_domain": "GENERIC",
                "to_domain": "TMA", "scope": "CTA", "regions": ["source"],
                "visibility_mask": {
                    "shared": True, "global_read": False, "global_write": False,
                    "tensor_read": False, "tensor_write": False, "control_only": False,
                },
            },
            {"id": "issue", "op": f"{prefix}.tma_store.issue", "resource": "stores", "group_owner": "owner", "source_region": "source", "destination_region": "destination", "bytes": 64},
            {"id": "commit", "op": f"{prefix}.tma_store.commit_group", "resource": "stores", "group_owner": "owner", "group_seq": 0},
            {"id": "wait", "op": f"{prefix}.tma_store.wait_group", "resource": "stores", "group_owner": "owner", "wait_group_count": 0, "wait_mode": "GLOBAL"},
            {"id": "read", "op": f"{prefix}.mem.read", "region": "destination"},
        ]
    snapshot["operations"] = [
        {"actor": "owner", "seq": sequence, **operation}
        for sequence, operation in enumerate(operations)
    ]
    return snapshot


def _without_wait(snapshot: dict) -> dict:
    snapshot["operations"] = [
        operation
        for operation in snapshot["operations"]
        if not operation["op"].endswith("mbarrier.wait_parity")
    ]
    read = next(
        operation
        for operation in snapshot["operations"]
        if operation["op"].endswith("mem.read") and operation["actor"] == "consumer"
    )
    read["seq"] = 2
    return snapshot


def _wrong_proxy_domain(snapshot: dict) -> dict:
    proxy = next(
        operation
        for operation in snapshot["operations"]
        if operation["op"].endswith("proxy_fence")
    )
    proxy["to_domain"] = "TMA"
    return snapshot


def _witness_shape(result: dict) -> dict:
    witness = result["witness"]
    return {
        "category": witness["category"],
        "effects": sorted(
            (effect["mode"], effect["region"], effect["window"][0] == effect["window"][1])
            for effect in witness["effects"]
        ),
        "missing_path": witness["missing_path"],
        "prefix_kinds": [step["kind"] for step in witness["reachable_execution_prefix"]],
    }


def test_both_frontend_operation_tables_cover_every_implemented_pgir_event():
    assert frozenset(TILELANG_EVENT_TYPES.values()) == IMPLEMENTED_EVENTS
    assert frozenset(TRITON_EVENT_TYPES.values()) == IMPLEMENTED_EVENTS


@pytest.mark.parametrize(
    ("path", "prefix", "stage", "convert"),
    [
        (TILELANG, "tilelang", "post-inject-fence-proxy", tilelang_snapshot_to_module),
        (TRITON, "triton", "post-ttgir-pipeline", triton_snapshot_to_module),
    ],
)
def test_v2_extraction_certificate_preserves_legacy_pgir_semantics(
    path: Path,
    prefix: str,
    stage: str,
    convert,
):
    legacy = convert(_load(path))
    certified = convert(_as_v2(_load(path), prefix, stage))
    assert semantic_signature(certified) == semantic_signature(legacy)


def test_certified_empty_protocol_graphs_are_equivalent_and_safe_across_frontends():
    modules = []
    for path, prefix, stage, convert in [
        (TILELANG, "tilelang", "post-inject-fence-proxy", tilelang_snapshot_to_module),
        (TRITON, "triton", "post-ttgir-pipeline", triton_snapshot_to_module),
    ]:
        snapshot = _as_v2(_load(path), prefix, stage)
        snapshot["resources"] = []
        snapshot["regions"] = []
        snapshot["operations"] = []
        snapshot["extraction"].update({
            "static_expansions": 0,
            "executable_nodes_seen": 0,
            "classified_nodes": 0,
            "emitting_nodes": 0,
            "elided_nodes": 0,
            "events_emitted": 0,
        })
        module = convert(snapshot)
        assert check_module(module, engine="maximal")["verdict"] == "SAFE"
        assert check_module(module, engine="crosscheck")["verdict"] == "SAFE"
        modules.append(module)
    assert semantic_signature(modules[0]) == semantic_signature(modules[1])


@pytest.mark.parametrize(
    ("path", "prefix", "stage", "convert"),
    [
        (TILELANG, "tilelang", "post-inject-fence-proxy", tilelang_snapshot_to_module),
        (TRITON, "triton", "post-ttgir-pipeline", triton_snapshot_to_module),
    ],
)
def test_v2_extraction_certificate_rejects_incomplete_accounting(
    path: Path,
    prefix: str,
    stage: str,
    convert,
):
    snapshot = _as_v2(_load(path), prefix, stage)
    snapshot["extraction"]["classified_nodes"] -= 1
    with pytest.raises(FrontendError) as caught:
        convert(snapshot)
    assert caught.value.code == "FRONTEND_EXTRACTION_ACCOUNTING_INVALID"


@pytest.mark.parametrize(
    ("path", "prefix", "stage", "convert"),
    [
        (TILELANG, "tilelang", "post-inject-fence-proxy", tilelang_snapshot_to_module),
        (TRITON, "triton", "post-ttgir-pipeline", triton_snapshot_to_module),
    ],
)
def test_v2_extraction_certificate_requires_event_evidence(
    path: Path,
    prefix: str,
    stage: str,
    convert,
):
    snapshot = _as_v2(_load(path), prefix, stage)
    snapshot["operations"][0].pop("evidence")
    with pytest.raises(FrontendError) as caught:
        convert(snapshot)
    assert caught.value.code == "FRONTEND_FIELDS_MISSING"


@pytest.mark.parametrize(
    ("path", "prefix", "stage", "convert"),
    [
        (TILELANG, "tilelang", "post-inject-fence-proxy", tilelang_snapshot_to_module),
        (TRITON, "triton", "post-ttgir-pipeline", triton_snapshot_to_module),
    ],
)
def test_v2_extraction_certificate_rejects_bad_ir_digest(
    path: Path,
    prefix: str,
    stage: str,
    convert,
):
    snapshot = _as_v2(_load(path), prefix, stage)
    snapshot["profile"]["source_ir_sha256"] = "not-a-digest"
    with pytest.raises(FrontendError) as caught:
        convert(snapshot)
    assert caught.value.code == "FRONTEND_PROVENANCE_INVALID"


def test_frontends_preserve_typed_async_fields_exactly():
    visibility_mask = {
        "shared": True,
        "global_read": False,
        "global_write": False,
        "tensor_read": True,
        "tensor_write": False,
        "control_only": False,
    }
    cases = [
        ("mbarrier.arrive_expect_tx", "MBarrierArriveExpectTx", {
            "resource": "bar", "lifecycle_epoch": 0, "generation": 2,
            "count": 4, "bytes": 128, "predicate": True,
        }),
        ("tma_load.issue", "TmaLoadIssue", {
            "resource": "bar", "source_region": "global", "destination_region": "shared",
            "bytes": 128, "lifecycle_epoch": 0, "generation": 2,
            "engine_id": "dma", "engine_seq": 3,
        }),
        ("cp_async.issue", "CpAsyncIssue", {
            "resource": "cp", "source_region": "global", "destination_region": "shared",
            "bytes": 16, "group_owner": "owner", "engine_id": "dma",
            "engine_seq": 4, "predicate": False,
        }),
        ("wgmma.issue", "WgmmaIssue", {
            "resource": "mma", "group_owner": "wg", "regions": ["a", "b"],
            "destination_region": "acc", "engine_id": "tensor", "engine_seq": 5,
        }),
        ("wgmma.wait_group", "WgmmaWaitGroup", {
            "resource": "mma", "group_owner": "wg", "wait_group_count": 1,
            "completion_kind": "OPERANDS_RELEASED",
        }),
        ("tma_store.issue", "TmaStoreIssue", {
            "resource": "store", "group_owner": "owner", "source_region": "shared",
            "destination_region": "global", "bytes": 128, "engine_id": "dma", "engine_seq": 6,
        }),
        ("tma_store.wait_group", "TmaStoreWaitGroup", {
            "resource": "store", "group_owner": "owner", "wait_group_count": 0,
            "wait_mode": "READ_ONLY",
        }),
        ("proxy_fence", "ProxyFence", {
            "from_domain": "GENERIC", "to_domain": "TENSOR_CORE", "scope": "CTA",
            "regions": ["a", "b"], "visibility_mask": visibility_mask,
        }),
    ]
    for suffix, event_type, fields in cases:
        common = {"id": "event", "actor": "owner", "seq": 0}
        tilelang = tilelang_event({**common, "op": f"tilelang.{suffix}", **fields}, 0)
        triton = triton_event({**common, "op": f"triton.{suffix}", **fields}, 0)
        assert tilelang == triton
        assert tilelang["type"] == event_type
        for field, value in fields.items():
            assert tilelang[field] == value


def test_equivalent_tilelang_and_triton_normalize_to_one_semantic_signature():
    tilelang = tilelang_snapshot_to_module(_load(TILELANG))
    triton = triton_snapshot_to_module(_load(TRITON))
    assert semantic_signature(tilelang) == semantic_signature(triton)
    assert semantic_digest(tilelang) == semantic_digest(triton)


def test_equivalent_safe_frontends_have_same_verdict_and_work_certificate():
    reports = [
        check_module(tilelang_snapshot_to_module(_load(TILELANG)), engine="crosscheck"),
        check_module(triton_snapshot_to_module(_load(TRITON)), engine="crosscheck"),
    ]
    assert [report["verdict"] for report in reports] == ["SAFE", "SAFE"]
    comparable = (
        "n_ext",
        "t_eff",
        "scheduler_work_units",
        "transition_attempts",
        "anchor_boundaries",
        "pdg_nodes",
        "pdg_edges",
        "pdg_edges_by_kind",
    )
    assert {
        key: reports[0]["statistics"][key]
        for key in comparable
    } == {
        key: reports[1]["statistics"][key]
        for key in comparable
    }


def test_equivalent_async_frontends_cover_proxy_group_engine_and_window_semantics():
    tilelang = tilelang_snapshot_to_module(_load(TILELANG_ASYNC))
    triton = triton_snapshot_to_module(_load(TRITON_ASYNC))
    assert semantic_signature(tilelang) == semantic_signature(triton)
    reports = [
        check_module(tilelang, engine="crosscheck"),
        check_module(triton, engine="crosscheck"),
    ]
    assert [report["verdict"] for report in reports] == ["SAFE", "SAFE"]
    assert reports[0]["artifacts"]["engine_completion_streams"] == reports[1]["artifacts"]["engine_completion_streams"]
    for report in reports:
        assert report["statistics"]["engine_completion_streams"] == 1
        assert report["statistics"]["group_completion_streams"] == 0
        assert report["statistics"]["full_enabled_scans"] == 0
        assert report["statistics"]["state_clones"] == 0
        assert report["statistics"]["replayed_prefix_steps"] == 0


@pytest.mark.parametrize("protocol", ["wgmma", "tma_store"])
def test_equivalent_typed_protocols_share_one_pgir_signature_and_checker(protocol: str):
    tilelang = tilelang_snapshot_to_module(
        _typed_protocol_snapshot(TILELANG_ASYNC, "tilelang", protocol)
    )
    triton = triton_snapshot_to_module(
        _typed_protocol_snapshot(TRITON_ASYNC, "triton", protocol)
    )
    assert semantic_signature(tilelang) == semantic_signature(triton)
    assert semantic_digest(tilelang) == semantic_digest(triton)
    reports = [
        check_module(tilelang, engine="crosscheck"),
        check_module(triton, engine="crosscheck"),
    ]
    assert [report["verdict"] for report in reports] == ["SAFE", "SAFE"]
    assert reports[0]["artifacts"]["crosscheck"] == reports[1]["artifacts"]["crosscheck"]


def test_equivalent_wrong_proxy_frontends_have_same_async_hazard_witness():
    tilelang = tilelang_snapshot_to_module(_wrong_proxy_domain(_load(TILELANG_ASYNC)))
    triton = triton_snapshot_to_module(_wrong_proxy_domain(_load(TRITON_ASYNC)))
    assert semantic_signature(tilelang) == semantic_signature(triton)
    reports = [
        check_module(tilelang, engine="crosscheck"),
        check_module(triton, engine="crosscheck"),
    ]
    assert [report["verdict"] for report in reports] == ["DATA_HAZARD", "DATA_HAZARD"]
    assert _witness_shape(reports[0]) == _witness_shape(reports[1])


def test_equivalent_missing_wait_frontends_have_same_hazard_witness_shape():
    tilelang = tilelang_snapshot_to_module(_without_wait(_load(TILELANG)))
    triton = triton_snapshot_to_module(_without_wait(_load(TRITON)))
    assert semantic_signature(tilelang) == semantic_signature(triton)
    reports = [
        check_module(tilelang, engine="crosscheck"),
        check_module(triton, engine="crosscheck"),
    ]
    assert [report["verdict"] for report in reports] == ["DATA_HAZARD", "DATA_HAZARD"]
    assert _witness_shape(reports[0]) == _witness_shape(reports[1])


def test_compare_cli_reports_cross_dsl_equivalence():
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "pipeguard",
            "compare",
            str(TILELANG),
            str(TRITON),
            "--left-frontend",
            "tilelang",
            "--right-frontend",
            "triton",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    document = json.loads(process.stdout)
    assert document["equivalent"] is True
    assert document["semantic_equivalent"] is True
    assert document["checker_equivalent"] is True
    assert document["left"]["semantic_sha256"] == document["right"]["semantic_sha256"]
    assert document["left"]["verdict"] == document["right"]["verdict"] == "SAFE"


def test_cross_dsl_signature_and_cli_are_deterministic():
    command = [
        sys.executable,
        "-m",
        "pipeguard",
        "compare",
        str(TILELANG),
        str(TRITON),
        "--left-frontend",
        "tilelang",
        "--right-frontend",
        "triton",
    ]
    first = subprocess.run(command, check=True, capture_output=True).stdout
    second = subprocess.run(command, check=True, capture_output=True).stdout
    assert first == second
