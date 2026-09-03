from __future__ import annotations

from itertools import product

from pipeguard.checker import check_module
from pipeguard.model import load_module, module_from_dict
from test_checker import (
    FIXTURES,
    channel_module,
    cp_async_module,
    tma_load_proxy_module,
    tma_store_module,
    wgmma_module,
)


def _assert_expected(module, expected: str, scenario: tuple) -> None:
    result = check_module(module, engine="crosscheck")
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
        expected_safe = (
            (not overwrite_operand or wait_operands)
            and (not read_accumulator or wait_result)
        )
        expected = "SAFE" if expected_safe else "DATA_HAZARD"
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

    for phases, control_only in product((1, 2), (False, True)):
        expected = "DATA_HAZARD" if control_only else "SAFE"
        _assert_expected(
            channel_module(phases=phases, control_only=control_only),
            expected,
            ("channel", phases, control_only),
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

    assert checked == 65
    assert safe > 0
    assert hazard > 0
