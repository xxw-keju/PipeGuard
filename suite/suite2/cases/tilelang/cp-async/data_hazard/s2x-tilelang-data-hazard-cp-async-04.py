# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {'case_id': 's2x-tilelang-data-hazard-cp-async-04', 'dsl': 'tilelang', 'result_class': 'DATA_HAZARD', 'primitive': 'cp.async', 'variant': 4, 'template_family': 'CP-WAIT', 'scenario_id': 'CP-WAIT', 'polarity': 'BUG', 'semantic_knobs': {'elements': 5, 'stages': 3, 'threads': 32, 'region_offset_bytes': 64, 'protocol_depth': 6, 'tilelang_tma_bytes': 80, 'triton_tma_block': 32, 'triton_tma_transfers': 5, 'wgmma_repeats': 5, 'wgmma_shape': [64, 64, 16], 'proxy_repeats': 5, 'proxy_store_shape': [64, 64]}, 'oracle': {'verdict': 'DATA_HAZARD', 'basis': 'independent paired protocol template plus variant-specific participants/region/depth'}, 'admitted': False, 'ai_adjudication': {'status': 'PENDING_INDEPENDENT_REVIEW', 'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order']}}

def pipeguard_resolve():
    old_shape = _cases._shape
    knobs = CASE_SPEC["semantic_knobs"]
    _cases._shape = lambda _variant: (knobs["elements"], knobs["stages"], knobs["threads"])
    try:
        if CASE_SPEC["primitive"] == "cp.async":
            return _cases._suite2_cp_async_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        if CASE_SPEC["primitive"] == "WGMMA":
            return _cases._suite2_wgmma_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        if CASE_SPEC["primitive"] == "Proxy":
            return _cases._suite2_proxy_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        return _cases.build_tilelang_case(
            CASE_SPEC["template_family"], CASE_SPEC["scenario_id"],
            CASE_SPEC["variant"], CASE_SPEC["polarity"],
        )
    finally:
        _cases._shape = old_shape
