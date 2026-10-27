# Redesigned executable Suite 2 DSL kernel (structural-diversity rework); do not hand edit.
from suite.frontend_pairs.sources import triton_cases as _cases

CASE_SPEC = {'admitted': False, 'ai_adjudication': {'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order'], 'status': 'PENDING_INDEPENDENT_REVIEW'}, 'case_id': 's2x-triton-sync-hazard-wgmma-02', 'dsl': 'triton', 'oracle': {'basis': 'structurally distinct per-variant protocol skeleton (suite2 diversity rework); hazard/safety grounded in post-pipeline TTGIR', 'verdict': 'SYNC_HAZARD'}, 'polarity': 'BUG', 'primitive': 'WGMMA', 'result_class': 'SYNC_HAZARD', 'scenario_id': 'WGMMA:TWO-GROUPS-NEWEST-CONSUMED-FIRST', 'semantic_knobs': {'elements': 3, 'protocol_depth': 4, 'region_offset_bytes': 32, 'stages': 4, 'threads': 96, 'tilelang_tma_bytes': 48, 'triton_tma_block': 32, 'triton_tma_transfers': 3, 'wgmma_repeats': 3, 'wgmma_shape': [64, 64, 16]}, 'template_family': 'WGMMA', 'variant': 2}

def pipeguard_resolve():
    s = CASE_SPEC
    if s["primitive"] == "cp.async":
        return _cases._s2v2_cp_async(s["variant"], s["result_class"])
    if s["primitive"] == "TMA":
        return _cases._s2v2_tma(s["variant"], s["result_class"])
    if s["primitive"] == "mbarrier":
        return _cases._s2v2_mbarrier_data_hazard(s["variant"])
    if s["primitive"] == "WGMMA":
        return _cases._s2v2_wgmma(s["variant"], s["result_class"])
    if s["primitive"] == "Proxy":
        return _cases._s2v2_proxy(s["variant"], s["result_class"])
    raise KeyError(f"unsupported primitive: {s['primitive']}")
