# Redesigned executable Suite 2 DSL kernel (structural-diversity rework); do not hand edit.
from suite.frontend_pairs.sources import triton_cases as _cases

CASE_SPEC = {'admitted': False, 'ai_adjudication': {'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order'], 'status': 'PENDING_INDEPENDENT_REVIEW'}, 'case_id': 's2x-triton-safe-proxy-00', 'dsl': 'triton', 'oracle': {'basis': 'structurally distinct per-variant protocol skeleton (suite2 diversity rework); hazard/safety grounded in post-pipeline TTGIR', 'verdict': 'SAFE'}, 'polarity': 'CORRECT', 'primitive': 'Proxy', 'result_class': 'SAFE', 'scenario_id': 'PROXY:WARP-SPECIALIZED-PUBLISH-FENCE-STORE', 'semantic_knobs': {'elements': 1, 'protocol_depth': 2, 'proxy_repeats': 1, 'proxy_store_shape': [64, 64], 'region_offset_bytes': 0, 'stages': 2, 'threads': 32, 'tilelang_tma_bytes': 16, 'triton_tma_block': 32, 'triton_tma_transfers': 1, 'wgmma_repeats': 1, 'wgmma_shape': [64, 64, 16]}, 'template_family': 'TMA-PROXY', 'variant': 0}

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
