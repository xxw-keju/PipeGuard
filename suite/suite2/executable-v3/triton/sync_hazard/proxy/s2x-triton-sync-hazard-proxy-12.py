# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import triton_cases as _cases

CASE_SPEC = {'case_id': 's2x-triton-sync-hazard-proxy-12', 'dsl': 'triton', 'result_class': 'SYNC_HAZARD', 'primitive': 'Proxy', 'variant': 12, 'template_family': 'WGMMA', 'scenario_id': 'WGMMA:MISSING-OPERAND-FENCE', 'polarity': 'BUG', 'semantic_knobs': {'elements': 13, 'stages': 2, 'threads': 32, 'region_offset_bytes': 192, 'protocol_depth': 14, 'tilelang_tma_bytes': 208, 'triton_tma_block': 224}, 'oracle': {'verdict': 'SYNC_HAZARD', 'basis': 'independent paired protocol template plus variant-specific participants/region/depth'}, 'admitted': False, 'ai_adjudication': {'status': 'PENDING_INDEPENDENT_REVIEW', 'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order']}}

def pipeguard_resolve():
    s = CASE_SPEC
    family, scenario = s["template_family"], s["scenario_id"]
    elements = s["semantic_knobs"]["elements"]
    threads = s["semantic_knobs"]["threads"]
    if family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        return _cases._memory_kernel(family, elements, threads)
    if family == "MB-REUSE": return _cases._mbarrier_kernel()
    if family in {"CP-WAIT", "CP-OVERLAP"}: return _cases._cp_async_kernel(family, threads)
    if family in {"TMA-OBSERVE", "TMA-PROXY"}: return _cases._tma_load_kernel(family, s["variant"])
    if family == "WGMMA-ENDPOINT": return _cases._wgmma_kernel()
    if family == "TMA-STORE-ENDPOINT": return _cases._tma_store_kernel()
    if family == "CTA-BARRIER": return _cases._named_cta_kernel(scenario, threads)
    if family == "MBARRIER": return _cases._named_mbarrier_kernel(scenario)
    if family == "CP-ASYNC": return _cases._named_cp_async_kernel(scenario, threads)
    if family == "TMA-LOAD": return _cases._named_tma_load_kernel(scenario)
    if family == "WGMMA": return _cases._named_wgmma_kernel(scenario)
    if family == "TMA-STORE": return _cases._named_tma_store_kernel(scenario)
    raise KeyError(f"unsupported template family: {family}")
