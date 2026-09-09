# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {'case_id': 's2x-tilelang-safe-wgmma-12', 'dsl': 'tilelang', 'result_class': 'SAFE', 'primitive': 'WGMMA', 'variant': 12, 'template_family': 'WGMMA-ENDPOINT', 'scenario_id': 'WGMMA-ENDPOINT', 'polarity': 'CORRECT', 'semantic_knobs': {'elements': 13, 'stages': 2, 'threads': 32, 'region_offset_bytes': 192, 'protocol_depth': 14, 'tilelang_tma_bytes': 208, 'triton_tma_block': 32, 'triton_tma_transfers': 13, 'wgmma_repeats': 13, 'wgmma_shape': [64, 64, 16]}, 'oracle': {'verdict': 'SAFE', 'basis': 'independent paired protocol template plus variant-specific participants/region/depth'}, 'admitted': False, 'ai_adjudication': {'status': 'PENDING_INDEPENDENT_REVIEW', 'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order']}}

def pipeguard_resolve():
    old_shape = _cases._shape
    knobs = CASE_SPEC["semantic_knobs"]
    _cases._shape = lambda _variant: (knobs["elements"], knobs["stages"], knobs["threads"])
    try:
        if CASE_SPEC["primitive"] == "WGMMA":
            return _cases._suite2_wgmma_case(
                CASE_SPEC["variant"], CASE_SPEC["result_class"]
            )
        return _cases.build_tilelang_case(
            CASE_SPEC["template_family"], CASE_SPEC["scenario_id"],
            CASE_SPEC["variant"], CASE_SPEC["polarity"],
        )
    finally:
        _cases._shape = old_shape
