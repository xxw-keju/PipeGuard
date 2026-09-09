from __future__ import annotations

from suite.pgir_direct.semantic_gates import evaluate_gates


def test_direct_semantic_gates_match_bounded_reference():
    payload = evaluate_gates()

    assert payload["gate_count"] == 21
    assert payload["all_equivalent"] is True
    assert all(
        record["direct_representation"] == "parametric-protocol"
        for record in payload["records"]
    )
    assert all(
        record["materialized_event_objects"] == 0
        for record in payload["records"]
    )
    assert all(
        record["reference_engine"] == "crosscheck"
        for record in payload["records"]
    )

    by_id = {record["gate_id"]: record for record in payload["records"]}
    assert by_id[
        "wgmma-eager-operand-completion-without-wait-stays-open"
    ]["direct_verdict"] == "DATA_HAZARD"
    assert by_id[
        "wgmma-operands-released-closes-only-operand-window"
    ]["direct_verdict"] == "SAFE"
    assert by_id[
        "tma-store-read-only-wait-does-not-make-global-visible"
    ]["direct_verdict"] == "DATA_HAZARD"
    assert by_id[
        "tma-store-global-complete-orders-global-read"
    ]["direct_verdict"] == "SAFE"
    assert by_id[
        "cp-async-held-completions-expose-capacity-overflow"
    ]["direct_diagnostic"] == "COMMIT_GROUP_CAPACITY_EXCEEDED"
    assert by_id[
        "cp-async-final-endpoint-wait-retires-capacity"
    ]["direct_verdict"] == "SAFE"
