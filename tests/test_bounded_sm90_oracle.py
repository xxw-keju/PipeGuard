from __future__ import annotations

import json
from itertools import product

import pytest

from pipeguard.checker import check_module
from pipeguard.model import PGIRError, load_module, module_from_dict
from test_checker import (
    FIXTURES,
    collective_handoff_module,
    cp_async_module,
    tma_load_proxy_module,
    tma_store_module,
    wgmma_module,
)


def _assert_expected(module, expected: str, scenario: tuple) -> None:
    result = check_module(module, engine="maximal")
    assert result["verdict"] == expected, (scenario, result)
    if expected != "SAFE":
        assert result["diagnostics"]
        assert result.get("witness")


def test_bounded_typed_completion_and_visibility_oracle():
    checked = 0
    safe = 0
    hazard = 0

    for group_count in (1, 2, 3):
        for wait_count in range(group_count + 1):
            for overwrite_group in range(group_count):
                completion_cutoff = group_count - wait_count
                expected = "SAFE" if overwrite_group < completion_cutoff else "DATA_HAZARD"
                _assert_expected(
                    cp_async_module(group_count, wait_count, overwrite_group),
                    expected,
                    ("cp_async", group_count, wait_count, overwrite_group),
                )
                checked += 1
                safe += expected == "SAFE"
                hazard += expected != "SAFE"

    for wait_operands, wait_result, overwrite_operand, read_accumulator in product(
        (False, True), repeat=4
    ):
        # Reading the accumulator without observing RESULT_READY is a protocol
        # violation (SYNC_HAZARD) and dominates; an operand overwrite without
        # the operand wait is an unordered conflict (DATA_HAZARD).
        if read_accumulator and not wait_result:
            expected = "SYNC_HAZARD"
        elif overwrite_operand and not wait_operands:
            expected = "DATA_HAZARD"
        else:
            expected = "SAFE"
        _assert_expected(
            wgmma_module(
                wait_operands=wait_operands,
                wait_result=wait_result,
                overwrite_operand=overwrite_operand,
                read_accumulator=read_accumulator,
            ),
            expected,
            (
                "wgmma",
                wait_operands,
                wait_result,
                overwrite_operand,
                read_accumulator,
            ),
        )
        checked += 1
        safe += expected == "SAFE"
        hazard += expected != "SAFE"

    for read_only_wait, global_wait, overwrite_source, read_destination in product(
        (False, True), repeat=4
    ):
        source_released = read_only_wait or global_wait
        expected_safe = (
            (not overwrite_source or source_released)
            and (not read_destination or global_wait)
        )
        expected = "SAFE" if expected_safe else "DATA_HAZARD"
        _assert_expected(
            tma_store_module(
                read_only_wait=read_only_wait,
                global_wait=global_wait,
                overwrite_source=overwrite_source,
                read_destination=read_destination,
            ),
            expected,
            (
                "tma_store",
                read_only_wait,
                global_wait,
                overwrite_source,
                read_destination,
            ),
        )
        checked += 1
        safe += expected == "SAFE"
        hazard += expected != "SAFE"

    for domain, fence_after_issue in product(
        ("TMA", "ASYNC_COPY", "TENSOR_CORE"),
        (False, True),
    ):
        expected = "SAFE" if domain == "TMA" and not fence_after_issue else "DATA_HAZARD"
        _assert_expected(
            module_from_dict(tma_load_proxy_module(domain, fence_after_issue)),
            expected,
            ("proxy", domain, fence_after_issue),
        )
        checked += 1
        safe += expected == "SAFE"
        hazard += expected != "SAFE"

    fixture_expectations = {
        "tma_completion_before_arrival.json": "SAFE",
        "tma_completion_after_arrival.json": "SAFE",
        "tma_bytes_mismatch.json": "SYNC_HAZARD",
    }
    for fixture, expected in fixture_expectations.items():
        _assert_expected(load_module(FIXTURES / fixture), expected, ("tma_load", fixture))
        checked += 1
        safe += expected == "SAFE"
        hazard += expected != "SAFE"

    assert checked == 61
    assert safe > 0
    assert hazard > 0


SM90_RULE_DISPOSITION = {
    "BAR-01": "validation",
    "BAR-02": "validation",
    "BAR-03": "partial-fail-closed",
    "BAR-04": "bounded",
    "MB-01": "validation",
    "MB-02": "partial-fail-closed",
    "MB-03": "validation",
    "MB-04": "partial-fail-closed",
    "MB-05": "unsupported",
    "MB-06": "partial-fail-closed",
    "MB-07": "bounded",
    "MB-08": "bounded",
    "MB-09": "bounded",
    "MB-10": "bounded",
    "CP-01": "bounded",
    "CP-02": "bounded",
    "CP-03": "bounded",
    "CP-04": "bounded",
    "TMA-01": "partial-fail-closed",
    "TMA-02": "partial-fail-closed",
    "TMA-03": "bounded",
    "TMA-04": "bounded",
    "TMA-05": "bounded",
    "TMA-06": "bounded",
    "WG-01": "partial-fail-closed",
    "WG-02": "validation",
    "WG-03": "bounded",
    "WG-04": "bounded",
    "WG-05": "partial-fail-closed",
    "WG-06": "bounded",
    "WG-07": "bounded",
    "WG-08": "bounded",
    "MEM-01": "bounded",
    "MEM-02": "partial-fail-closed",
    "MEM-03": "partial-fail-closed",
    "MEM-04": "bounded",
}


def test_every_sm90_rule_has_an_explicit_validation_disposition():
    expected = {
        *(f"BAR-{index:02d}" for index in range(1, 5)),
        *(f"MB-{index:02d}" for index in range(1, 11)),
        *(f"CP-{index:02d}" for index in range(1, 5)),
        *(f"TMA-{index:02d}" for index in range(1, 7)),
        *(f"WG-{index:02d}" for index in range(1, 9)),
        *(f"MEM-{index:02d}" for index in range(1, 5)),
    }
    assert set(SM90_RULE_DISPOSITION) == expected
    assert set(SM90_RULE_DISPOSITION.values()) == {
        "bounded",
        "validation",
        "partial-fail-closed",
        "unsupported",
    }


def _collective_module(arrive_mask: tuple[bool, bool], observe_mask: tuple[bool, bool]):
    raw = json.loads(json.dumps(collective_handoff_module().raw))
    raw["regions"] = []
    raw["resources"][0]["visibility_mask"] = {
        "shared": False,
        "global_read": False,
        "global_write": False,
        "tensor_read": False,
        "tensor_write": False,
        "control_only": True,
    }
    events = []
    for index, actor in enumerate(("a_writer", "b_reader")):
        sequence = 0
        if arrive_mask[index]:
            events.append({
                "id": f"arrive_{index}", "type": "BarrierArrive", "actor": actor,
                "seq": sequence, "resource": "sync", "count": 1, "invocation_id": 0,
            })
            sequence += 1
        if observe_mask[index]:
            events.append({
                "id": f"observe_{index}", "type": "BarrierObserve", "actor": actor,
                "seq": sequence, "resource": "sync", "invocation_id": 0,
            })
    raw["events"] = events
    return module_from_dict(raw)


def test_bounded_cta_barrier_oracle():
    checked = 0
    for arrive_mask in product((False, True), repeat=2):
        for observe_mask in product((False, True), repeat=2):
            any_event = any(arrive_mask) or any(observe_mask)
            expected = "SAFE" if all(arrive_mask) or not any_event else "SYNC_HAZARD"
            _assert_expected(
                _collective_module(arrive_mask, observe_mask),
                expected,
                ("cta_barrier", arrive_mask, observe_mask),
            )
            checked += 1
    assert checked == 16


def _point_conflict_module(
    left_offset: int,
    right_offset: int,
    left_write: bool,
    right_write: bool,
):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["actors"] = [
        {"id": "left", "kind": "LANE", "participants": [0]},
        {"id": "right", "kind": "LANE", "participants": [1]},
    ]
    raw["target"]["cta_threads"] = 2
    raw["resources"] = []
    raw["regions"] = [
        {"id": "left_region", "allocation_id": "shared", "address_space": "SHARED", "offset": left_offset, "extent": 4},
        {"id": "right_region", "allocation_id": "shared", "address_space": "SHARED", "offset": right_offset, "extent": 4},
    ]
    raw["events"] = [
        {"id": "left_access", "type": "MemWrite" if left_write else "MemRead", "actor": "left", "seq": 0, "region": "left_region"},
        {"id": "right_access", "type": "MemWrite" if right_write else "MemRead", "actor": "right", "seq": 0, "region": "right_region"},
    ]
    return module_from_dict(raw)


def test_bounded_point_conflict_oracle():
    checked = 0
    for left_offset, right_offset, left_write, right_write in product(
        (0, 4, 8),
        (0, 4, 8),
        (False, True),
        (False, True),
    ):
        overlap = max(left_offset, right_offset) < min(left_offset + 4, right_offset + 4)
        expected = "DATA_HAZARD" if overlap and (left_write or right_write) else "SAFE"
        _assert_expected(
            _point_conflict_module(left_offset, right_offset, left_write, right_write),
            expected,
            ("point_conflict", left_offset, right_offset, left_write, right_write),
        )
        checked += 1
    assert checked == 36


@pytest.mark.parametrize(
    ("event_type", "rule_id"),
    [
        ("MBarrierArriveDrop", "MB-05"),
        ("MBarrierTxComplete", "MB-06"),
    ],
)
def test_unimplemented_mbarrier_events_fail_closed(event_type: str, rule_id: str):
    raw = json.loads((FIXTURES / "safe_handoff.json").read_text())
    raw["events"] = [{
        "id": "unsupported", "type": event_type, "actor": "a_producer", "seq": 0,
    }]
    with pytest.raises(PGIRError) as caught:
        module_from_dict(raw)
    assert caught.value.code == "KNOWN_EVENT_UNSUPPORTED", rule_id
