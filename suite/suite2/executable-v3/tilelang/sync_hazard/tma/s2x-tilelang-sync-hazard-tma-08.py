# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import tilelang_cases as _cases

CASE_SPEC = {'case_id': 's2x-tilelang-sync-hazard-tma-08', 'dsl': 'tilelang', 'result_class': 'SYNC_HAZARD', 'primitive': 'TMA', 'variant': 8, 'template_family': 'TMA-STORE', 'scenario_id': 'TMA-STORE:STORE-WAIT-BEFORE-COMMIT', 'polarity': 'BUG', 'semantic_knobs': {'elements': 9, 'stages': 4, 'threads': 32, 'region_offset_bytes': 128, 'protocol_depth': 10, 'tilelang_tma_bytes': 144, 'triton_tma_block': 160}, 'oracle': {'verdict': 'SYNC_HAZARD', 'basis': 'independent paired protocol template plus variant-specific participants/region/depth'}, 'admitted': False, 'ai_adjudication': {'status': 'PENDING_INDEPENDENT_REVIEW', 'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order']}}

def pipeguard_resolve():
    old_shape = _cases._shape
    knobs = CASE_SPEC["semantic_knobs"]
    _cases._shape = lambda _variant: (knobs["elements"], knobs["stages"], knobs["threads"])
    try:
        return _cases.build_tilelang_case(
            CASE_SPEC["template_family"], CASE_SPEC["scenario_id"],
            CASE_SPEC["variant"], CASE_SPEC["polarity"],
        )
    finally:
        _cases._shape = old_shape
