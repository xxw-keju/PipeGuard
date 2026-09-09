from __future__ import annotations

import json
import subprocess
import sys
from copy import deepcopy

import pytest

from pipeguard.checker import check_module
from pipeguard.direct_parametric import NumericPDG
from pipeguard.model import PGIRError, module_from_dict
from pipeguard.parametric import (
    parametric_module_from_dict,
    stable_collective_event_id,
    stable_collective_point_id,
    stable_event_instance_id,
    stable_stream_instance_id,
)
from suite.pgir_direct.cases.catalog import CASES


def _visibility_mask() -> dict:
    return {
        "shared": True,
        "global_read": False,
        "global_write": False,
        "tensor_read": False,
        "tensor_write": False,
        "control_only": False,
    }


def parametric_memory_module() -> dict:
    return {
        "schema_version": "pgir-parametric-v1",
        "producer": {
            "dsl": "TRITON",
            "dsl_version": "test",
            "dsl_commit": "test",
            "backend_commit": "test",
            "adapter_version": "triton-pgir-parametric-v1",
            "command": "test",
        },
        "target": {
            "architecture": "sm90",
            "warp_size": 32,
            "cta_threads": 2,
            "shared_memory_bytes": 16,
            "grid_scope": "single_cta",
        },
        "assumptions": {
            "finite_control_flow": True,
            "statically_known_participants": True,
            "fair_async_completion": True,
            "no_cluster_effects": True,
            "alias_precision": "EXACT",
            "parity_normalization_status": "RESOLVED",
        },
        "control_structure": {"finite_event_graph": True},
        "owners": [{
            "id": "partition:0",
            "kind": "PARTITION",
            "participants": [0, 1],
        }],
        "program_streams": [{
            "id": "lane",
            "kind": "LANE",
            "instances": [0, 1],
        }],
        "resources": [],
        "regions": [{
            "id": "shared",
            "allocation_id": "smem",
            "address_space": "SHARED",
            "offset": 0,
            "extent": 4,
        }],
        "event_templates": [
            {
                "id": "write",
                "type": "MemWrite",
                "stream": "lane",
                "order": 0,
                "execution_scope": "INSTANCE",
                "instances": [0],
                "fields": {"region": "shared"},
            },
            {
                "id": "read",
                "type": "MemRead",
                "stream": "lane",
                "order": 1,
                "execution_scope": "INSTANCE",
                "instances": [1],
                "fields": {"region": "shared"},
            },
        ],
        "frontend_provenance": {
            "extraction_complete": True,
            "source": "parametric-test",
        },
        "frontend_evidence": {
            "participant_mapping": {
                "status": "RESOLVED",
                "method": "static-cta-geometry",
                "source": "fixture",
            },
            "lane_ownership": {
                "status": "RESOLVED",
                "method": "blocked-layout-inversion",
                "source": "fixture",
            },
            "predicate_mapping": {
                "status": "RESOLVED",
                "method": "static-mask-evaluation",
                "source": "fixture",
            },
            "alias": {
                "status": "EXACT",
                "method": "allocation-plus-byte-interval",
                "source": "fixture",
            },
        },
        "unsupported_constructs": [],
    }


def test_partition_owner_does_not_serialize_logical_lane_streams():
    compact = module_from_dict(parametric_memory_module())
    result = check_module(compact, engine="crosscheck")
    module = compact.expand_reference()

    assert result["verdict"] == "DATA_HAZARD"
    assert len(module.actors) == 3
    assert len(module.events) == 2
    assert compact.logical_stream_count == 2
    assert compact.template_count == 2
    assert result["statistics"]["materialized_event_objects"] == 0
    assert {event.actor for event in module.events} == {
        stable_stream_instance_id("lane", 0),
        stable_stream_instance_id("lane", 1),
    }
    assert all(event.execution_owner == event.actor for event in module.events)


def test_default_parametric_memory_check_does_not_expand(monkeypatch):
    compact = module_from_dict(parametric_memory_module())

    def forbidden(_self):
        raise AssertionError("default parameterized memory check expanded events")

    monkeypatch.setattr(type(compact), "expand_reference", forbidden)
    result = check_module(compact, engine="maximal")

    assert result["verdict"] == "DATA_HAZARD"
    assert result["statistics"]["logical_program_streams"] == 2
    assert result["statistics"]["materialized_event_objects"] == 0
    assert result["statistics"]["materialized_artifact_events"] == 0
    assert "canonical_trace" not in result["artifacts"]


def test_overlapping_all_read_instances_remain_compact_and_safe(monkeypatch):
    raw = parametric_memory_module()
    raw["target"]["cta_threads"] = 128
    raw["owners"][0]["participants"] = list(range(128))
    raw["program_streams"][0]["instances"] = list(range(128))
    raw["event_templates"] = [{
        "id": "read",
        "type": "MemRead",
        "stream": "lane",
        "order": 0,
        "execution_scope": "INSTANCE",
        "instances": list(range(128)),
        "fields": {"region": "shared"},
    }]
    compact = module_from_dict(raw)

    def forbidden(_self):
        raise AssertionError("default parameterized memory check expanded events")

    monkeypatch.setattr(type(compact), "expand_reference", forbidden)
    result = check_module(compact)

    assert result["verdict"] == "SAFE"
    assert result["statistics"]["events"] == 128
    assert result["statistics"]["event_templates"] == 1
    assert result["statistics"]["materialized_event_objects"] == 0


def test_same_logical_lane_keeps_program_order():
    raw = parametric_memory_module()
    raw["event_templates"][1]["instances"] = [0]
    compact = module_from_dict(raw)

    result = check_module(compact, engine="crosscheck")

    assert result["verdict"] == "SAFE"
    first = stable_event_instance_id("write", 0)
    second = stable_event_instance_id("read", 0)
    assert {
        "from": first,
        "to": second,
        "kind": "PROGRAM_ORDER",
    } in result["artifacts"]["pdg"]["edges"]


def test_collective_reference_expansion_rendezvous_once_and_orders_lanes():
    raw = parametric_memory_module()
    raw["resources"] = [{
        "id": "sync",
        "kind": "cta_barrier",
        "participants": ["partition:0"],
        "capacity": 2,
        "scope": "CTA",
        "visibility_mask": _visibility_mask(),
    }]
    raw["event_templates"] = [
        raw["event_templates"][0],
        {
            "id": "arrive",
            "type": "BarrierArrive",
            "stream": "lane",
            "order": 1,
            "execution_scope": "COLLECTIVE",
            "instances": [0, 1],
            "owner": "partition:0",
            "fields": {"resource": "sync", "count": 2, "invocation_id": 0},
        },
        {
            "id": "observe",
            "type": "BarrierObserve",
            "stream": "lane",
            "order": 2,
            "execution_scope": "COLLECTIVE",
            "instances": [0, 1],
            "owner": "partition:0",
            "fields": {"resource": "sync", "invocation_id": 0},
        },
        {
            **raw["event_templates"][1],
            "order": 3,
        },
    ]
    compact = module_from_dict(raw)
    result = check_module(compact, engine="crosscheck")
    module = compact.expand_reference()

    assert result["verdict"] == "SAFE"
    arrive = stable_collective_event_id("arrive")
    observe = stable_collective_event_id("observe")
    assert sum(event.id == arrive for event in module.events) == 1
    assert sum(event.id == observe for event in module.events) == 1
    assert module.event_map[arrive].execution_owner == "partition:0"
    for instance in (0, 1):
        assert stable_collective_point_id("arrive", instance, "enter") in module.event_map
        assert stable_collective_point_id("arrive", instance, "exit") in module.event_map


def test_elected_tma_source_event_is_not_cloned():
    raw = parametric_memory_module()
    raw["resources"] = [{
        "id": "bar",
        "kind": "mbarrier",
        "participants": ["partition:0"],
    }]
    raw["regions"] = [
        {
            "id": "global",
            "allocation_id": "global",
            "address_space": "GLOBAL",
            "offset": 0,
            "extent": 4,
        },
        raw["regions"][0],
    ]
    raw["event_templates"] = [{
        "id": "issue",
        "type": "TmaLoadIssue",
        "stream": "lane",
        "order": 0,
        "execution_scope": "ELECTED",
        "instances": [0],
        "owner": "partition:0",
        "fields": {
            "resource": "bar",
            "source_region": "global",
            "destination_region": "shared",
            "bytes": 4,
            "lifecycle_epoch": 0,
            "generation": 0,
        },
    }]
    compact = parametric_module_from_dict(raw)
    module = compact.expand_reference()

    issues = [event for event in module.events if event.type == "TmaLoadIssue"]
    assert len(issues) == 1
    assert issues[0].actor == stable_stream_instance_id("lane", 0)
    assert issues[0].execution_owner == "partition:0"


def test_parametric_certainty_claims_require_structured_evidence():
    raw = parametric_memory_module()
    raw["frontend_evidence"]["lane_ownership"] = {
        "status": "UNRESOLVED",
        "method": "layout-guess",
        "source": "fixture",
    }

    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)

    assert caught.value.code == "PARAMETRIC_EVIDENCE_UNRESOLVED"
    assert caught.value.details["claim"] == "lane_ownership"


def test_parametric_source_and_synthetic_namespaces_are_disjoint():
    raw = parametric_memory_module()
    raw["event_templates"][0]["id"] = "completion:forged"
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "RESERVED_ID_NAMESPACE"


def test_parametric_cli_validates_normalizes_and_checks_without_expansion(tmp_path):
    path = tmp_path / "compact.pgir.json"
    path.write_text(json.dumps(parametric_memory_module()))

    validate = subprocess.run(
        [sys.executable, "-m", "pipeguard", "validate", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    validation = json.loads(validate.stdout)
    assert validation["valid"] is True
    assert validation["schema_version"] == "pgir-parametric-v1"
    assert validation["counts"] == {
        "actors": 2,
        "events": 2,
        "regions": 1,
        "resources": 0,
    }

    normalize = subprocess.run(
        [sys.executable, "-m", "pipeguard", "normalize", str(path)],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(normalize.stdout)["schema_version"] == "pgir-parametric-v1"

    check = subprocess.run(
        [sys.executable, "-m", "pipeguard", "check", str(path)],
        capture_output=True,
        text=True,
    )
    assert check.returncode == 1
    report = json.loads(check.stdout)
    assert report["verdict"] == "DATA_HAZARD"
    assert report["statistics"]["materialized_event_objects"] == 0


def test_parametric_input_is_not_mutated_by_reference_expansion():
    raw = parametric_memory_module()
    before = json.loads(json.dumps(raw))
    module_from_dict(raw)
    assert raw == before


def test_memory_template_dependencies_use_general_parametric_checker():
    raw = parametric_memory_module()
    raw["event_templates"][1]["dependencies"] = [{
        "template": "write",
        "relation": "ALL_INSTANCES",
    }]
    result = check_module(module_from_dict(raw), engine="maximal")
    assert result["verdict"] == "DATA_HAZARD"
    assert result["statistics"]["representation"] == "parametric-protocol"

def test_numeric_pdg_indexes_synthetic_reachability_once():
    graph = NumericPDG(parametric_module_from_dict(parametric_memory_module()))
    first = graph.synthetic("first")
    second = graph.synthetic("second")
    graph.add_edge(0, first, "TEST")
    graph.add_edge(first, second, "TEST")
    graph.add_edge(second, 1, "TEST")

    assert graph.reachable(first, 1)
    initial_visits = graph.reachability_edge_visits
    assert graph.reachable(first, second)
    assert graph.reachability_edge_visits == initial_visits
    assert graph.reachability_cache_hits == 1
    assert len(graph._synthetic_indices or {}) == 2

    third = graph.synthetic("third")
    graph.add_edge(1, third, "TEST")
    assert graph._synthetic_ancestors is None
    assert graph.reachable(first, third)
    assert graph.reachability_edge_visits > initial_visits

def test_parametric_resource_can_name_a_stream_instance_subset():
    raw = parametric_memory_module()
    raw["resources"] = [{
        "id": "cp:lane:1",
        "kind": "async_group",
        "participants": [{"stream": "lane", "instances": [1]}],
        "capacity": 8,
        "protocol": "CP_ASYNC",
        "scope": "CTA",
    }]
    module = module_from_dict(raw)
    expanded = module.expand_reference()
    assert expanded.resources[0].participants == (
        stable_stream_instance_id("lane", 1),
    )

def test_parametric_resource_rejects_out_of_range_stream_instance():
    raw = parametric_memory_module()
    raw["resources"] = [{
        "id": "cp:bad",
        "kind": "async_group",
        "participants": [{"stream": "lane", "instances": [2]}],
        "capacity": 8,
        "protocol": "CP_ASYNC",
        "scope": "CTA",
    }]
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "PARAMETRIC_INSTANCE_OUT_OF_RANGE"

def _single_stream_parametric(raw: dict) -> dict:
    assert len(raw["actors"]) == 1
    source_actor = raw["actors"][0]["id"]
    owner_id = "execution-owner"
    resources = deepcopy(raw["resources"])
    for resource in resources:
        resource["participants"] = [
            owner_id if participant == source_actor else participant
            for participant in resource["participants"]
        ]

    templates = []
    for event in sorted(raw["events"], key=lambda item: item["seq"]):
        fields = {
            key: deepcopy(value)
            for key, value in event.items()
            if key not in {"id", "type", "actor", "seq", "depends_on", "owner"}
        }
        if fields.get("group_owner") == source_actor:
            fields["group_owner"] = owner_id
        if "parity_evidence" in fields:
            fields["parity_evidence"]["logical_stream_id"] = "$self"
        if event["type"].startswith("Wgmma"):
            execution_scope = "COLLECTIVE"
        elif event["type"] == "TmaLoadIssue":
            execution_scope = "ELECTED"
        else:
            execution_scope = "INSTANCE"
        template = {
            "id": event["id"],
            "type": event["type"],
            "stream": "lane",
            "order": event["seq"],
            "execution_scope": execution_scope,
            "instances": [0],
            "fields": fields,
        }
        if event["type"] not in {"MemRead", "MemWrite"}:
            template["owner"] = owner_id
        if event.get("depends_on"):
            template["dependencies"] = [
                {"template": dependency, "relation": "SAME_INSTANCE"}
                for dependency in event["depends_on"]
            ]
        templates.append(template)

    return {
        "schema_version": "pgir-parametric-v1",
        "producer": deepcopy(raw["producer"]),
        "target": deepcopy(raw["target"]),
        "assumptions": deepcopy(raw["assumptions"]),
        "control_structure": deepcopy(raw["control_structure"]),
        "owners": [{
            "id": owner_id,
            "kind": raw["actors"][0]["kind"],
            "participants": (
                deepcopy(raw["actors"][0]["participants"])
                if raw["actors"][0]["kind"] == "WARPGROUP"
                else [0]
            ),
        }],
        "program_streams": [{
            "id": "lane",
            "kind": "LANE",
            "instances": [0],
        }],
        "resources": resources,
        "regions": deepcopy(raw["regions"]),
        "event_templates": templates,
        "frontend_provenance": {
            "extraction_complete": True,
            "source": "direct-case-parametric-adapter",
        },
        "frontend_evidence": {
            claim: {
                "status": "RESOLVED" if claim != "alias" else "EXACT",
                "method": "direct-case-single-stream",
                "source": "fixture",
            }
            for claim in (
                "participant_mapping",
                "lane_ownership",
                "predicate_mapping",
                "alias",
            )
        },
        "unsupported_constructs": [],
    }

def test_five_axis_once_without_rendezvous_keeps_independent_lane_crossings():
    raw = parametric_memory_module()
    raw["event_templates"] = [{
        "id": "publish",
        "type": "ProxyFence",
        "stream": "lane",
        "order": 0,
        "program_instances": [0, 1],
        "executors": [0],
        "transition": "ONCE",
        "rendezvous": "NONE",
        "fields": {
            "from_domain": "GENERIC",
            "to_domain": "TMA",
            "scope": "CTA",
            "regions": ["shared"],
            "visibility_mask": _visibility_mask(),
        },
    }]
    compact = module_from_dict(raw)

    assert compact.lane_trace_offsets == (0, 1, 2)
    assert compact.lane_trace_template_ids == (0, 0)
    assert compact.templates[0].program_instances == (0, 1)
    assert compact.templates[0].executors == (0,)
    assert compact.templates[0].transition == "ONCE"
    assert compact.templates[0].rendezvous == "NONE"

    result = check_module(compact, engine="maximal")
    assert result["verdict"] == "SAFE"
    assert result["statistics"]["frontier_visits"] == 2
    assert result["statistics"]["protocol_transitions"] == 1
    assert result["statistics"]["collective_arrivals"] == 0

def _two_lane_mbarrier_reuse(*, ordered: bool) -> dict:
    raw = parametric_memory_module()
    raw["resources"] = [{
        "id": "bar",
        "kind": "mbarrier",
        "participants": [{"stream": "lane", "instances": [0, 1]}],
    }]
    common = {
        "stream": "lane",
        "program_instances": [0, 1],
        "transition": "ONCE",
        "rendezvous": "NONE",
    }
    raw["event_templates"] = [
        {
            **common,
            "id": "init",
            "type": "MBarrierInit",
            "order": 0,
            "executors": [0],
            "fields": {"resource": "bar", "lifecycle_epoch": 0, "count": 1},
        },
        {
            **common,
            "id": "arrive0",
            "type": "MBarrierArrive",
            "order": 1,
            "executors": [0],
            "fields": {
                "resource": "bar", "lifecycle_epoch": 0,
                "generation": 0, "count": 1, "predicate": True,
            },
        },
        {
            **common,
            "id": "wait0",
            "type": "MBarrierWaitParity",
            "order": 2,
            "executors": [1],
            "fields": {
                "resource": "bar", "lifecycle_epoch": 0,
                "generation": 0, "raw_parity": 0,
                "parity_evidence": {
                    "derivation_kind": "PHASE_SSA",
                    "logical_stream_id": "lane:1:bar",
                    "lifecycle_epoch": 0,
                    "stream_rank": 0,
                    "polling_coalesced": True,
                },
            },
        },
        {
            **common,
            "id": "arrive1",
            "type": "MBarrierArrive",
            "order": 3,
            "executors": [0],
            "fields": {
                "resource": "bar", "lifecycle_epoch": 0,
                "generation": 1, "count": 1, "predicate": True,
            },
            **({
                "dependencies": [{"template": "wait0", "relation": "COLLECTIVE"}]
            } if ordered else {}),
        },
    ]
    return raw

def test_once_without_rendezvous_rejects_unordered_phase_reuse():
    report = check_module(module_from_dict(_two_lane_mbarrier_reuse(ordered=False)))
    assert report["verdict"] == "SYNC_HAZARD"
    assert report["diagnostics"][0]["code"] == "INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE"


def test_sync_hazard_counts_declared_conflicts_after_protocol_error():
    raw = _two_lane_mbarrier_reuse(ordered=False)
    raw["event_templates"].extend((
        {
            "id": "write_after_error",
            "type": "MemWrite",
            "stream": "lane",
            "order": 4,
            "program_instances": [0],
            "executors": [0],
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "fields": {"region": "shared"},
        },
        {
            "id": "read_after_error",
            "type": "MemRead",
            "stream": "lane",
            "order": 5,
            "program_instances": [1],
            "executors": [1],
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "fields": {"region": "shared"},
        },
    ))

    report = check_module(module_from_dict(raw), engine="maximal")

    assert report["verdict"] == "SYNC_HAZARD"
    assert report["diagnostics"][0]["code"] == (
        "INVALID_MBARRIER_UNOBSERVED_PHASE_REUSE"
    )
    statistics = report["statistics"]
    assert statistics["exact_conflicting_access_pairs"] == 1
    assert statistics["race_candidate_pairs"] == 1
    assert statistics["pairs_examined_before_termination"] == 0
    assert statistics["exact_r_complete"] is True
    assert statistics["race_scan_scope"] == (
        "DECLARED_ACCESS_CONFLICT_ENUMERATION"
    )
    assert statistics["race_ordering_performed"] is False
    assert statistics["race_scan_s"] > 0.0



def test_once_transition_dependency_allows_ordered_phase_reuse():
    report = check_module(module_from_dict(_two_lane_mbarrier_reuse(ordered=True)))
    assert report["verdict"] == "SAFE"
    assert report["statistics"]["protocol_transitions"] == 4

def _shared_chain_mbarrier_reuse(size: int) -> dict:
    raw = _two_lane_mbarrier_reuse(ordered=True)
    init, arrive0, wait0, arrive1 = raw["event_templates"]
    init["fields"]["count"] = size
    arrive0["fields"]["count"] = size
    chain = []
    predecessor = "wait0"
    for index in range(size):
        identifier = f"fence_{index}"
        chain.append({
            "id": identifier,
            "type": "ProxyFence",
            "stream": "lane",
            "order": 3 + index,
            "program_instances": [0, 1],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "dependencies": [{"template": predecessor, "relation": "COLLECTIVE"}],
            "fields": {
                "from_domain": "GENERIC",
                "to_domain": "TMA",
                "scope": "CTA",
                "regions": ["shared"],
                "visibility_mask": _visibility_mask(),
            },
        })
        predecessor = identifier
    arrivals = []
    for index in range(size):
        event = deepcopy(arrive1)
        event["id"] = f"arrive1_{index}"
        event["order"] = 3 + size + index
        event["fields"]["count"] = 1
        event["dependencies"] = [{
            "template": predecessor,
            "relation": "COLLECTIVE",
        }]
        arrivals.append(event)
    raw["event_templates"] = [init, arrive0, wait0, *chain, *arrivals]
    return raw


def test_wait_last_shared_chain_has_linear_scheduler_work():
    size = 64
    compact = module_from_dict(_shared_chain_mbarrier_reuse(size))
    report = check_module(compact, engine="maximal")

    assert report["verdict"] == "SAFE"
    statistics = report["statistics"]
    assert statistics["protocol_transitions"] == 2 * size + 3
    assert statistics["completion_transitions"] == 2
    assert statistics["deferred_waits"] >= 1
    assert statistics["frontier_visits"] <= 3 * statistics["logical_events"]
    assert statistics["reachability_queries"] == 0
    assert statistics["materialized_event_objects"] == 0


def _deferred_tma_phase_reuse(*, ordered: bool) -> dict:
    raw = parametric_memory_module()
    raw["program_streams"][0]["instances"] = [0]
    raw["owners"][0]["participants"] = [0]
    raw["target"]["cta_threads"] = 1
    raw["resources"] = [{
        "id": "bar",
        "kind": "mbarrier",
        "participants": ["partition:0"],
    }]
    raw["regions"] = [
        {"id": "source0", "allocation_id": "g0", "address_space": "GLOBAL", "offset": 0, "extent": 16},
        {"id": "tile0", "allocation_id": "s0", "address_space": "SHARED", "offset": 0, "extent": 16},
        {"id": "source1", "allocation_id": "g1", "address_space": "GLOBAL", "offset": 0, "extent": 16},
        {"id": "tile1", "allocation_id": "s1", "address_space": "SHARED", "offset": 0, "extent": 16},
    ]
    common = {
        "stream": "lane",
        "program_instances": [0],
        "executors": [0],
        "transition": "ONCE",
        "rendezvous": "NONE",
        "owner": "partition:0",
    }
    def wait(identifier: str, order: int, generation: int) -> dict:
        return {
            **common,
            "id": identifier,
            "type": "MBarrierWaitParity",
            "order": order,
            "fields": {
                "resource": "bar",
                "lifecycle_epoch": 0,
                "generation": generation,
                "raw_parity": generation & 1,
                "parity_evidence": {
                    "derivation_kind": "PHASE_SSA",
                    "logical_stream_id": "lane:0:bar",
                    "lifecycle_epoch": 0,
                    "stream_rank": generation,
                    "polling_coalesced": True,
                },
            },
        }
    events = [
        {**common, "id": "init", "type": "MBarrierInit", "order": 0,
         "fields": {"resource": "bar", "lifecycle_epoch": 0, "count": 1}},
        {**common, "id": "expect0", "type": "MBarrierExpectTx", "order": 1,
         "fields": {"resource": "bar", "lifecycle_epoch": 0, "generation": 0, "bytes": 16}},
        {**common, "id": "issue0", "type": "TmaLoadIssue", "order": 2,
         "fields": {"resource": "bar", "lifecycle_epoch": 0, "generation": 0, "bytes": 16,
                    "source_region": "source0", "destination_region": "tile0"}},
        {**common, "id": "arrive0", "type": "MBarrierArrive", "order": 3,
         "fields": {"resource": "bar", "lifecycle_epoch": 0, "generation": 0, "count": 1, "predicate": True}},
    ]
    generation1 = [
        {**common, "id": "expect1", "type": "MBarrierExpectTx", "order": 0,
         "fields": {"resource": "bar", "lifecycle_epoch": 0, "generation": 1, "bytes": 16}},
        {**common, "id": "issue1", "type": "TmaLoadIssue", "order": 0,
         "fields": {"resource": "bar", "lifecycle_epoch": 0, "generation": 1, "bytes": 16,
                    "source_region": "source1", "destination_region": "tile1"}},
        {**common, "id": "arrive1", "type": "MBarrierArrive", "order": 0,
         "fields": {"resource": "bar", "lifecycle_epoch": 0, "generation": 1, "count": 1, "predicate": True}},
        wait("wait1", 0, 1),
    ]
    ordered_events = [*events, wait("wait0", 0, 0), *generation1] if ordered else [
        *events, *generation1[:2], wait("wait0", 0, 0), *generation1[2:]
    ]
    for order, event in enumerate(ordered_events):
        event["order"] = order
    raw["event_templates"] = ordered_events
    return raw


@pytest.mark.parametrize("ordered", [False, True])
def test_tma_completion_last_exposes_early_next_generation(ordered: bool):
    compact = module_from_dict(_deferred_tma_phase_reuse(ordered=ordered))
    direct = check_module(compact, engine="maximal")
    reference = check_module(compact.expand_reference(), engine="maximal")

    assert direct["verdict"] == reference["verdict"]
    assert (direct.get("diagnostics") or [{}])[0].get("code") == (
        reference.get("diagnostics") or [{}]
    )[0].get("code")
    assert direct["statistics"]["exact_conflicting_access_pairs"] == 0
    assert direct["statistics"]["race_candidate_pairs"] == 0
    if ordered:
        assert direct["verdict"] == "SAFE"
        assert direct["statistics"]["completion_transitions"] == 4
    else:
        assert direct["verdict"] == "SYNC_HAZARD"
        assert direct["diagnostics"][0]["code"] == "EARLY_NEXT_GENERATION"
        assert direct["statistics"]["completion_transitions"] == 0


@pytest.mark.parametrize("label", [
    "PGIR-MB-08-NAMED-TWO-GENERATION-HANDOFF-CORRECT",
    "PGIR-CP-04-NAMED-WAIT-FRONTIER-CORRECT",
    "PGIR-TMA-04-TMA-OBSERVE-V00-CORRECT",
    "PGIR-WG-07-WG-08-NAMED-DUAL-ENDPOINT-CORRECT",
    "PGIR-TMA-05-NAMED-STORE-DUAL-ENDPOINT-CORRECT",
    "PGIR-MB-08-MB-REUSE-V00-BUG",
    "PGIR-CP-04-CP-WAIT-V00-BUG",
    "PGIR-TMA-04-TMA-OBSERVE-V00-BUG",
    "PGIR-WG-08-WGMMA-ENDPOINT-V00-BUG",
    "PGIR-TMA-05-TMA-STORE-ENDPOINT-V00-BUG",
])
def test_parametric_protocol_maximal_does_not_expand_and_matches_reference(
    label: str,
    monkeypatch: pytest.MonkeyPatch,
):
    case = next(item for item in CASES if item.label == label)
    compact = module_from_dict(_single_stream_parametric(case.raw))
    expected = check_module(compact.expand_reference(), engine="maximal")

    def forbidden(_self):
        raise AssertionError("default parameterized protocol check expanded events")

    monkeypatch.setattr(type(compact), "expand_reference", forbidden)
    actual = check_module(compact, engine="maximal")

    assert actual["verdict"] == expected["verdict"]
    actual_diagnostics = actual.get("diagnostics") or [{}]
    expected_diagnostics = expected.get("diagnostics") or [{}]
    assert actual_diagnostics[0].get("code") == expected_diagnostics[0].get("code")
    assert actual["statistics"]["representation"] == "parametric-protocol"
    assert actual["statistics"]["materialized_event_objects"] == 0
    if "witness" in expected:
        assert actual["witness"]["category"] == expected["witness"]["category"]
        assert actual["witness"].get("events") == expected["witness"].get("events")
        if "effects" in expected["witness"]:
            def semantic_effects(report):
                return sorted(
                    (
                        effect.get("actor"),
                        effect.get("mode"),
                        effect.get("region"),
                    )
                    for effect in report["witness"]["effects"]
                )

            assert semantic_effects(actual) == semantic_effects(expected)
            assert all(
                effect.get("id") and len(effect.get("window", [])) == 2
                for effect in actual["witness"]["effects"]
            )

def test_parametric_collective_uses_implicit_pdg_and_lazy_anchors(
    monkeypatch: pytest.MonkeyPatch,
):
    raw = parametric_memory_module()
    instances = list(range(32))
    raw["target"]["cta_threads"] = len(instances)
    raw["owners"][0]["participants"] = instances
    raw["program_streams"][0]["instances"] = instances
    raw["resources"] = [{
        "id": "sync",
        "kind": "cta_barrier",
        "participants": ["partition:0"],
        "capacity": len(instances),
        "scope": "CTA",
        "visibility_mask": _visibility_mask(),
    }]
    raw["event_templates"] = [
        {
            "id": "write",
            "type": "MemWrite",
            "stream": "lane",
            "order": 0,
            "execution_scope": "INSTANCE",
            "instances": [0],
            "fields": {"region": "shared"},
        },
        {
            "id": "arrive",
            "type": "BarrierArrive",
            "stream": "lane",
            "order": 1,
            "execution_scope": "COLLECTIVE",
            "instances": instances,
            "owner": "partition:0",
            "fields": {
                "resource": "sync",
                "count": len(instances),
                "invocation_id": 0,
            },
        },
        {
            "id": "observe",
            "type": "BarrierObserve",
            "stream": "lane",
            "order": 2,
            "execution_scope": "COLLECTIVE",
            "instances": instances,
            "owner": "partition:0",
            "fields": {"resource": "sync", "invocation_id": 0},
        },
        {
            "id": "read",
            "type": "MemRead",
            "stream": "lane",
            "order": 3,
            "execution_scope": "INSTANCE",
            "instances": [1],
            "fields": {"region": "shared"},
        },
    ]
    compact = module_from_dict(raw)

    def forbidden(_self):
        raise AssertionError("parameterized collective check expanded events")

    monkeypatch.setattr(type(compact), "expand_reference", forbidden)
    result = check_module(compact, engine="maximal")

    assert result["verdict"] == "SAFE"
    assert result["statistics"]["representation"] == "parametric-protocol"
    assert result["statistics"]["materialized_event_objects"] == 0
    assert result["statistics"]["pdg_resident_program_nodes"] == 0
    assert result["statistics"]["pdg_implicit_program_edges"] > 0
    assert result["statistics"]["pdg_implicit_boundary_edges"] > 0
    assert result["artifacts"]["pdg"]["representation"] == "parametric"
    assert "nodes" not in result["artifacts"]["pdg"]
    assert result["artifacts"]["boundaries"] == []
    assert result["artifacts"]["parametric_program"]["materialized_boundaries"] == 0
    assert result["artifacts"]["parametric_program"]["logical_boundaries"] > 0

def test_all_single_stream_direct_cases_use_parameterized_execution():
    checked = 0
    for case in CASES:
        if case.expected_stage != "CHECKER" or len(case.raw["actors"]) != 1:
            continue
        report = check_module(
            module_from_dict(_single_stream_parametric(case.raw)),
            engine="maximal",
        )
        diagnostic = (
            report["diagnostics"][0]["code"]
            if report.get("diagnostics")
            else None
        )
        assert report["verdict"] == case.expected_verdict, case.label
        assert diagnostic == case.expected_diagnostic, case.label
        checked += 1

    assert checked == 195


def _many_independent_tma_barriers(size: int) -> dict:
    raw = parametric_memory_module()
    raw["program_streams"][0]["instances"] = [0]
    raw["owners"][0]["participants"] = [0]
    raw["target"]["cta_threads"] = 1
    raw["target"]["shared_memory_bytes"] = 16 * size
    raw["resources"] = [
        {"id": f"bar{index}", "kind": "mbarrier", "participants": ["partition:0"]}
        for index in range(size)
    ]
    raw["regions"] = []
    templates = []
    for index in range(size):
        source = f"source{index}"
        tile = f"tile{index}"
        raw["regions"].extend((
            {"id": source, "allocation_id": f"g{index}", "address_space": "GLOBAL", "offset": 0, "extent": 16},
            {"id": tile, "allocation_id": f"s{index}", "address_space": "SHARED", "offset": 0, "extent": 16},
        ))
        common = {
            "stream": "lane",
            "program_instances": [0],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "owner": "partition:0",
        }
        base = 4 * index
        templates.extend((
            {**common, "id": f"init{index}", "type": "MBarrierInit", "order": base,
             "fields": {"resource": f"bar{index}", "lifecycle_epoch": 0, "count": 1}},
            {**common, "id": f"expect{index}", "type": "MBarrierExpectTx", "order": base + 1,
             "fields": {"resource": f"bar{index}", "lifecycle_epoch": 0, "generation": 0, "bytes": 16}},
            {**common, "id": f"issue{index}", "type": "TmaLoadIssue", "order": base + 2,
             "fields": {"resource": f"bar{index}", "lifecycle_epoch": 0, "generation": 0, "bytes": 16,
                        "source_region": source, "destination_region": tile}},
            {**common, "id": f"arrive{index}", "type": "MBarrierArrive", "order": base + 3,
             "fields": {"resource": f"bar{index}", "lifecycle_epoch": 0, "generation": 0,
                        "count": 1, "predicate": True}},
        ))
    raw["event_templates"] = templates
    return raw


def test_completion_ready_queue_is_linear_across_many_tma_barriers():
    size = 64
    report = check_module(
        module_from_dict(_many_independent_tma_barriers(size)),
        engine="maximal",
    )

    assert report["verdict"] == "SAFE"
    statistics = report["statistics"]
    assert statistics["protocol_transitions"] == 4 * size
    assert statistics["completion_transitions"] == 2 * size
    assert statistics["completion_queue_pushes"] == 2 * size
    assert statistics["completion_queue_pops"] == 2 * size
    assert statistics["frontier_visits"] == 4 * size
    assert statistics["exact_conflicting_access_pairs"] == 0
    assert statistics["materialized_event_objects"] == 0


@pytest.mark.parametrize(
    ("expected_bytes", "expected_verdict", "expected_diagnostic"),
    [
        (16, "SYNC_HAZARD", "TX_BYTES_MISMATCH"),
        (32, "SAFE", None),
    ],
)
def test_terminal_mbarrier_transaction_bytes_are_validated(
    expected_bytes: int,
    expected_verdict: str,
    expected_diagnostic: str | None,
):
    raw = _many_independent_tma_barriers(1)
    raw["target"]["shared_memory_bytes"] = 32
    for region in raw["regions"]:
        region["extent"] = 32
    raw["event_templates"][1]["fields"]["bytes"] = expected_bytes
    raw["event_templates"][2]["fields"]["bytes"] = 32

    compact = module_from_dict(raw)
    direct = check_module(compact, engine="maximal")
    reference = check_module(compact.expand_reference(), engine="maximal")
    direct_diagnostic = (
        direct["diagnostics"][0]["code"] if direct.get("diagnostics") else None
    )
    reference_diagnostic = (
        reference["diagnostics"][0]["code"]
        if reference.get("diagnostics")
        else None
    )

    assert direct["verdict"] == reference["verdict"] == expected_verdict
    assert direct_diagnostic == reference_diagnostic == expected_diagnostic
    assert direct["statistics"]["exact_conflicting_access_pairs"] == 0
    assert direct["statistics"]["race_candidate_pairs"] == 0
    if expected_bytes == 16:
        assert direct["statistics"]["completion_queue_pushes"] == 1
        assert direct["statistics"]["completion_queue_pops"] == 1


def _many_sequential_cp_async_waits(size: int) -> dict:
    case = next(
        case
        for case in CASES
        if case.family == "CP-WAIT"
        and case.variant == 0
        and case.polarity == "CORRECT"
    )
    raw = deepcopy(case.raw)
    raw["regions"] = []
    raw["events"] = []
    for index in range(size):
        source = f"source{index}"
        tile = f"tile{index}"
        raw["regions"].extend((
            {
                "id": source,
                "allocation_id": f"global{index}",
                "address_space": "GLOBAL",
                "offset": 0,
                "extent": 4,
            },
            {
                "id": tile,
                "allocation_id": f"shared{index}",
                "address_space": "SHARED",
                "offset": 0,
                "extent": 4,
            },
        ))
        base = 3 * index
        raw["events"].extend((
            {
                "id": f"issue{index}",
                "type": "CpAsyncIssue",
                "actor": "owner",
                "seq": base,
                "resource": "copies",
                "source_region": source,
                "destination_region": tile,
                "bytes": 4,
                "group_owner": "owner",
            },
            {
                "id": f"commit{index}",
                "type": "CpAsyncCommitGroup",
                "actor": "owner",
                "seq": base + 1,
                "resource": "copies",
                "group_owner": "owner",
                "group_seq": index,
            },
            {
                "id": f"wait{index}",
                "type": "CpAsyncWaitGroup",
                "actor": "owner",
                "seq": base + 2,
                "resource": "copies",
                "group_owner": "owner",
                "wait_group_count": 0,
            },
        ))
    return _single_stream_parametric(raw)


@pytest.mark.parametrize("size", [8, 32, 128])
def test_group_wait_frontiers_advance_linearly(size: int):
    compact = module_from_dict(_many_sequential_cp_async_waits(size))

    direct = check_module(compact, engine="maximal")
    reference = check_module(compact.expand_reference(), engine="maximal")

    assert direct["verdict"] == reference["verdict"] == "SAFE"
    statistics = direct["statistics"]
    assert statistics["exact_conflicting_access_pairs"] == 0
    assert statistics["group_completion_frontier_advances"] == size
    assert statistics["group_wait_observation_advances"] == size
    assert statistics["group_wait_readiness_checks"] <= 2 * size
    assert statistics["materialized_event_objects"] == 0


def _disjoint_nonmonotone_writes(size: int) -> dict:
    raw = parametric_memory_module()
    effect_count = 2 * size
    raw["target"]["shared_memory_bytes"] = 4 * effect_count
    raw["regions"] = []
    raw["event_templates"] = []
    for lane in (0, 1):
        for index in range(size):
            rank = lane * size + index
            offset = ((rank * 37) % effect_count) * 4
            region = f"slice{rank}"
            raw["regions"].append({
                "id": region,
                "allocation_id": "shared",
                "address_space": "SHARED",
                "offset": offset,
                "extent": 4,
            })
            raw["event_templates"].append({
                "id": f"write{lane}_{index}",
                "type": "MemWrite",
                "stream": "lane",
                "order": rank,
                "execution_scope": "INSTANCE",
                "instances": [lane],
                "fields": {"region": region},
            })
    return raw


@pytest.mark.parametrize("size", [32, 128, 512])
def test_fixed_pass_race_ordering_is_linear_for_zero_output(size: int):
    compact = module_from_dict(_disjoint_nonmonotone_writes(size))

    direct = check_module(compact, engine="maximal")
    reference = check_module(compact.expand_reference(), engine="maximal")

    assert direct["verdict"] == reference["verdict"] == "SAFE"
    statistics = direct["statistics"]
    effects = 2 * size
    assert statistics["exact_conflicting_access_pairs"] == 0
    assert statistics["race_ordering_effect_visits"] == 90 * effects
    assert statistics["race_sweep_insertions"] == effects
    assert statistics["race_sweep_expirations"] <= effects
    assert statistics["race_sweep_candidate_visits"] == 0
    assert (
        statistics["race_ordering_effect_visits"]
        + statistics["race_sweep_insertions"]
        + statistics["race_sweep_expirations"]
        <= 92 * effects
    )


def test_parametric_region_rejects_64_bit_address_overflow():
    raw = parametric_memory_module()
    raw["regions"][0]["offset"] = (1 << 64) - 1
    raw["regions"][0]["extent"] = 2

    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)

    assert caught.value.code == "PARAMETRIC_REGION_ADDRESS_OUT_OF_RANGE"



def _two_lane_tma_store_wait(wait_mode: str) -> dict:
    raw = parametric_memory_module()
    raw["owners"][0]["participants"] = [0, 1]
    raw["resources"] = [{
        "id": "store",
        "kind": "async_group",
        "participants": ["partition:0"],
        "capacity": 8,
        "protocol": "TMA_STORE",
        "scope": "CTA",
    }]
    raw["regions"] = [
        {
            "id": "source",
            "allocation_id": "shared:source",
            "address_space": "SHARED",
            "offset": 0,
            "extent": 16,
        },
        {
            "id": "output",
            "allocation_id": "global:output",
            "address_space": "GLOBAL",
            "offset": 0,
            "extent": 16,
        },
    ]
    common = {
        "stream": "lane",
        "program_instances": [0, 1],
        "executors": [0],
        "transition": "ONCE",
        "rendezvous": "NONE",
    }
    raw["event_templates"] = [
        {
            **common,
            "id": "issue",
            "type": "TmaStoreIssue",
            "order": 0,
            "fields": {
                "resource": "store",
                "group_owner": "partition:0",
                "source_region": "source",
                "destination_region": "output",
                "bytes": 16,
            },
        },
        {
            **common,
            "id": "commit",
            "type": "TmaStoreCommitGroup",
            "order": 1,
            "fields": {
                "resource": "store",
                "group_owner": "partition:0",
                "group_seq": 0,
            },
        },
        {
            **common,
            "id": "wait",
            "type": "TmaStoreWaitGroup",
            "order": 2,
            "fields": {
                "resource": "store",
                "group_owner": "partition:0",
                "wait_group_count": 0,
                "wait_mode": wait_mode,
            },
        },
        {
            "id": "read",
            "type": "MemRead",
            "stream": "lane",
            "order": 3,
            "program_instances": [1],
            "executors": [1],
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "fields": {"region": "output"},
        },
    ]
    return raw


@pytest.mark.parametrize(
    ("wait_mode", "expected"),
    [("GLOBAL", "SAFE"), ("READ_ONLY", "DATA_HAZARD")],
)
def test_tma_store_wait_publishes_completion_to_all_program_lanes(
    wait_mode: str,
    expected: str,
):
    report = check_module(
        module_from_dict(_two_lane_tma_store_wait(wait_mode)),
        engine="maximal",
    )

    assert report["verdict"] == expected
    assert report["statistics"]["exact_conflicting_access_pairs"] == 1


def _proxy_dependency_handoff(*, ordered: bool) -> dict:
    raw = parametric_memory_module()
    raw["owners"][0]["participants"] = [0]
    raw["resources"] = [{
        "id": "bar",
        "kind": "mbarrier",
        "participants": ["partition:0"],
    }]
    raw["regions"] = [
        {
            "id": "source",
            "allocation_id": "global:source",
            "address_space": "GLOBAL",
            "offset": 0,
            "extent": 16,
        },
        {
            "id": "tile",
            "allocation_id": "shared:tile",
            "address_space": "SHARED",
            "offset": 0,
            "extent": 16,
        },
    ]
    dependency = (
        [{"template": "producer", "relation": "ALL_INSTANCES"}]
        if ordered else []
    )
    raw["event_templates"] = [
        {
            "id": "producer",
            "type": "MemWrite",
            "stream": "lane",
            "order": 0,
            "program_instances": [1],
            "executors": [1],
            "transition": "PER_INSTANCE",
            "rendezvous": "NONE",
            "fields": {"region": "source"},
        },
        {
            "id": "init",
            "type": "MBarrierInit",
            "stream": "lane",
            "order": 1,
            "program_instances": [0],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "fields": {"resource": "bar", "lifecycle_epoch": 0, "count": 1},
        },
        {
            "id": "expect",
            "type": "MBarrierExpectTx",
            "stream": "lane",
            "order": 2,
            "program_instances": [0],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "fields": {
                "resource": "bar", "lifecycle_epoch": 0,
                "generation": 0, "bytes": 16,
            },
        },
        {
            "id": "proxy",
            "type": "ProxyFence",
            "stream": "lane",
            "order": 3,
            "program_instances": [0],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "dependencies": dependency,
            "fields": {
                "from_domain": "GENERIC", "to_domain": "TMA",
                "scope": "CTA", "regions": ["source"],
                "visibility_mask": _visibility_mask(),
            },
        },
        {
            "id": "issue",
            "type": "TmaLoadIssue",
            "stream": "lane",
            "order": 4,
            "program_instances": [0],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "fields": {
                "resource": "bar", "lifecycle_epoch": 0, "generation": 0,
                "source_region": "source", "destination_region": "tile",
                "bytes": 16,
            },
        },
        {
            "id": "arrive",
            "type": "MBarrierArrive",
            "stream": "lane",
            "order": 5,
            "program_instances": [0],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "fields": {
                "resource": "bar", "lifecycle_epoch": 0,
                "generation": 0, "count": 1,
            },
        },
        {
            "id": "wait",
            "type": "MBarrierWaitParity",
            "stream": "lane",
            "order": 6,
            "program_instances": [0],
            "executors": [0],
            "transition": "ONCE",
            "rendezvous": "NONE",
            "fields": {
                "resource": "bar", "lifecycle_epoch": 0,
                "generation": 0, "raw_parity": 0,
                "parity_evidence": {
                    "derivation_kind": "BARRIER_HISTORY",
                    "logical_stream_id": "partition:0:bar",
                    "lifecycle_epoch": 0, "stream_rank": 0,
                    "polling_coalesced": True,
                },
            },
        },
    ]
    return raw


@pytest.mark.parametrize(
    ("ordered", "expected"),
    [(True, "SAFE"), (False, "DATA_HAZARD")],
)
def test_proxy_fence_honors_all_instances_dependency_from_other_lane(
    ordered: bool,
    expected: str,
):
    compact = module_from_dict(_proxy_dependency_handoff(ordered=ordered))

    direct = check_module(compact, engine="maximal")

    assert direct["verdict"] == expected
    assert direct["statistics"]["exact_conflicting_access_pairs"] == 1


@pytest.mark.parametrize(
    ("fragments", "expected"),
    [
        (((0, 112, "same"),), "SAFE"),
        (((0, 56, "same"), (56, 56, "same")), "SAFE"),
        (((40, 72, "same"), (0, 56, "same")), "SAFE"),
        (((0, 56, "same"), (57, 55, "same")), "DATA_HAZARD"),
        (((0, 56, "same"), (56, 56, "other")), "DATA_HAZARD"),
    ],
)
def test_parametric_proxy_region_union_exactly_covers_async_source(
    fragments,
    expected,
):
    raw = deepcopy(next(
        case.raw
        for case in CASES
        if case.label == "PGIR-TMA-06-NAMED-PROXY-AND-WAIT-CORRECT"
    ))
    source = next(region for region in raw["regions"] if region["id"] == "source")
    fragment_ids = []
    for index, (offset, extent, allocation_id) in enumerate(fragments):
        identifier = f"source_fragment_{index}"
        raw["regions"].append({
            **source,
            "id": identifier,
            "allocation_id": (
                source["allocation_id"]
                if allocation_id == "same"
                else allocation_id
            ),
            "offset": offset,
            "extent": extent,
        })
        fragment_ids.append(identifier)
    next(event for event in raw["events"] if event["id"] == "proxy")["regions"] = fragment_ids

    compact = module_from_dict(_single_stream_parametric(raw))
    direct = check_module(compact, engine="maximal")
    reference = check_module(compact.expand_reference(), engine="crosscheck")

    assert direct["verdict"] == reference["verdict"] == expected
    assert direct["statistics"]["exact_conflicting_access_pairs"] == 1



def _many_proxy_fences(size: int) -> dict:
    raw = deepcopy(next(
        case.raw
        for case in CASES
        if case.label == "PGIR-TMA-06-NAMED-PROXY-AND-WAIT-CORRECT"
    ))
    original = {event["id"]: event for event in raw["events"]}
    source = next(region for region in raw["regions"] if region["id"] == "source")
    raw["regions"].append({
        **source,
        "id": "source_short",
        "extent": source["extent"] - 1,
    })
    early = size // 2
    events = []
    for index in range(early):
        fence = deepcopy(original["proxy"])
        fence["id"] = f"early_covering_fence_{index}"
        events.append(fence)
    for index in range(size):
        events.append({
            "id": f"source_write_{index}",
            "type": "MemWrite",
            "actor": "owner",
            "region": "source",
        })
    events.extend((deepcopy(original["init"]), deepcopy(original["expect"])))
    for index in range(early, size):
        fence = deepcopy(original["proxy"])
        fence["id"] = f"late_noncovering_fence_{index}"
        fence["regions"] = ["source_short"]
        events.append(fence)
    events.extend((
        deepcopy(original["proxy"]),
        deepcopy(original["issue"]),
        deepcopy(original["arrive"]),
        deepcopy(original["wait"]),
    ))
    for sequence, event in enumerate(events):
        event["seq"] = sequence
    raw["events"] = events
    return _single_stream_parametric(raw)


@pytest.mark.parametrize("size", [4, 16, 64, 128])
def test_proxy_fence_candidate_index_is_linear(size: int):
    compact = module_from_dict(_many_proxy_fences(size))

    direct = check_module(compact, engine="maximal")
    reference = check_module(compact.expand_reference(), engine="crosscheck")

    assert direct["verdict"] == reference["verdict"] == "SAFE"
    statistics = direct["statistics"]
    logical_events = 2 * size + 6
    assert statistics["exact_conflicting_access_pairs"] == size
    assert statistics["proxy_fence_index_queries"] == size
    assert statistics["proxy_fence_index_build_visits"] <= 200 * logical_events
    assert (
        statistics["proxy_fence_index_build_visits"]
        + statistics["proxy_fence_index_queries"]
        <= 201 * logical_events
    )
