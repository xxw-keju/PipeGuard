# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {'case_id': 's2x-tilelang-safe-tma-13', 'dsl': 'tilelang', 'result_class': 'SAFE', 'primitive': 'TMA', 'variant': 13, 'template_family': 'TMA-OBSERVE', 'scenario_id': 'TMA-OBSERVE', 'polarity': 'CORRECT', 'semantic_knobs': {'elements': 14, 'stages': 3, 'threads': 64, 'region_offset_bytes': 208, 'protocol_depth': 15, 'tilelang_tma_bytes': 224, 'triton_tma_block': 32, 'triton_tma_transfers': 14, 'wgmma_repeats': 14, 'wgmma_shape': [64, 64, 16], 'proxy_repeats': 14, 'proxy_store_shape': [64, 64]}, 'oracle': {'verdict': 'SAFE', 'basis': 'independent paired protocol template plus variant-specific participants/region/depth'}, 'admitted': False, 'ai_adjudication': {'status': 'PENDING_INDEPENDENT_REVIEW', 'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order']}}

def pipeguard_resolve():
    old_shape = _cases._shape
    knobs = CASE_SPEC["semantic_knobs"]
    _cases._shape = lambda _variant: (knobs["elements"], knobs["stages"], knobs["threads"])
    try:
        if CASE_SPEC["primitive"] == "WGMMA":
            return _cases._suite2_wgmma_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        if CASE_SPEC["primitive"] == "Proxy":
            return _cases._suite2_proxy_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        if CASE_SPEC["primitive"] == "TMA" and CASE_SPEC["result_class"] == "SYNC_HAZARD":
            return _cases._suite2_tma_sync_case(CASE_SPEC["variant"])
        return _cases.build_tilelang_case(
            CASE_SPEC["template_family"], CASE_SPEC["scenario_id"],
            CASE_SPEC["variant"], CASE_SPEC["polarity"],
        )
    finally:
        _cases._shape = old_shape
