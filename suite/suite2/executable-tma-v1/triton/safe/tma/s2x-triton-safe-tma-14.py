# Generated executable Suite 2 DSL kernel; do not hand edit.
from suite.frontend_pairs.sources import triton_cases as _cases

CASE_SPEC = {'case_id': 's2x-triton-safe-tma-14', 'dsl': 'triton', 'result_class': 'SAFE', 'primitive': 'TMA', 'variant': 14, 'template_family': 'TMA-OBSERVE', 'scenario_id': 'TMA-OBSERVE', 'polarity': 'CORRECT', 'semantic_knobs': {'elements': 15, 'stages': 4, 'threads': 96, 'region_offset_bytes': 224, 'protocol_depth': 16, 'tilelang_tma_bytes': 240, 'triton_tma_block': 32, 'triton_tma_transfers': 15, 'wgmma_repeats': 15, 'wgmma_shape': [64, 64, 16], 'proxy_repeats': 15, 'proxy_store_shape': [64, 64]}, 'oracle': {'verdict': 'SAFE', 'basis': 'independent paired protocol template plus variant-specific participants/region/depth'}, 'admitted': False, 'ai_adjudication': {'status': 'PENDING_INDEPENDENT_REVIEW', 'required_evidence': ['source_and_lowered_operations', 'physical_participants', 'resource_generation_or_endpoint', 'byte_overlap_if_applicable', 'reachable_prefix', 'missing_or_present_typed_order']}}

def pipeguard_resolve():
    s = CASE_SPEC
    family, scenario = s["template_family"], s["scenario_id"]
    elements = s["semantic_knobs"]["elements"]
    threads = s["semantic_knobs"]["threads"]
    if s["primitive"] == "WGMMA":
        return _cases._suite2_wgmma_kernel(s["variant"], s["result_class"])
    if s["primitive"] == "Proxy":
        return _cases._suite2_proxy_kernel(s["variant"], s["result_class"])
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
