from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TritonFaultStrategy:
    suffix: str
    rule_id: str
    family: str
    name: str
    description: str
    helper: str
    scratch_type: str
    auxiliary_type: str
    adjudication_class: str
    adjudication_status: str
    adjudication_method: str
    adjudication_rationale: str
    kernel_ids: tuple[str, ...] | None = None
    rejected_kernel_ids: tuple[str, ...] = ()
    rejection_rationale: str | None = None


_STRATEGIES = (
    TritonFaultStrategy(
        suffix="m8-global-roundtrip",
        rule_id="MUT-M8-GLOBAL-RAW-ROUNDTRIP",
        family="GLOBAL_MEMORY_ORDER",
        name="M8 mixed-width global RAW round trip",
        description=(
            "Add an output-neutral global store/load round trip whose independent Coalesce layouts "
            "create a cross-thread RAW dependency without a CTA barrier."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    pipeguard_rows = tl.arange(0, 8)[:, None]
    pipeguard_cols = tl.arange(0, 256)[None, :]
    pipeguard_loaded = tl.load(
        pipeguard_wide + pipeguard_slot * 2048 + pipeguard_rows * 256 + pipeguard_cols
    )
    pipeguard_aliases = (
        pipeguard_scratch + pipeguard_slot * 2048 + pipeguard_rows * 256 + pipeguard_cols
    )
    pipeguard_zero = tl.zeros((8, 256), tl.float32)
    tl.store(pipeguard_aliases, pipeguard_zero)
    pipeguard_reloaded = tl.load(pipeguard_aliases)
    pipeguard_values = pipeguard_reloaded + pipeguard_loaded.to(tl.float32) * 0.0
    return tl.max(tl.max(pipeguard_values, axis=1), axis=0)
''',
        scratch_type="*fp32",
        auxiliary_type="*fp16",
        adjudication_class="DATA_HAZARD",
        adjudication_status="CONFIRMED",
        adjudication_method="MANUAL_PROTOCOL_TRACE",
        adjudication_rationale=(
            "The store and reload use the same unique logical addresses, but Coalesce assigns different "
            "thread layouts. Eleven of twenty workloads mismatch; inserting tl.debug_barrier between "
            "the operations restores all twenty. PipeGuard reports DATA_HAZARD/UNORDERED_CONFLICT."
        ),
        rejected_kernel_ids=("triton-dense-gemm",),
        rejection_rationale=(
            "Final TTGIR/PTX/SASS and exact lane-byte audit for Dense GEMM show 2,048 same-lane "
            "producer/consumer overlaps and zero cross-lane overlaps. The original cross-thread RAW "
            "label is rejected; numeric mismatch remains a separate observation."
        ),
    ),
    TritonFaultStrategy(
        suffix="m9-masked-store-roundtrip",
        rule_id="MUT-M9-GLOBAL-RAW-MASKED-STORE",
        family="GLOBAL_MEMORY_ORDER",
        name="M9 runtime-true masked producer",
        description=(
            "Add an output-neutral global round trip with a runtime all-true mask on the producer store, "
            "forcing incompatible store/load layouts without changing source semantics."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    pipeguard_rows = tl.arange(0, 8)[:, None]
    pipeguard_cols = tl.arange(0, 256)[None, :]
    pipeguard_offsets = pipeguard_slot * 2048 + pipeguard_rows * 256 + pipeguard_cols
    pipeguard_mask = tl.load(pipeguard_wide + pipeguard_offsets) != 0
    pipeguard_aliases = pipeguard_scratch + pipeguard_offsets
    pipeguard_zero = tl.zeros((8, 256), tl.float32)
    tl.store(pipeguard_aliases, pipeguard_zero, mask=pipeguard_mask)
    pipeguard_reloaded = tl.load(pipeguard_aliases)
    return tl.max(tl.max(pipeguard_reloaded, axis=1), axis=0)
''',
        scratch_type="*fp32",
        auxiliary_type="*i8",
        adjudication_class="DATA_HAZARD",
        adjudication_status="CONFIRMED",
        adjudication_method="MANUAL_PROTOCOL_TRACE",
        adjudication_rationale=(
            "The runtime mask buffer is all true, so every unique address is overwritten with zero before "
            "it is reloaded. All twenty workloads mismatch without a barrier and pass with a barrier."
        ),
        rejected_kernel_ids=("triton-dense-gemm",),
        rejection_rationale=(
            "Final TTGIR/PTX/SASS and exact lane-byte audit for Dense GEMM show 2,048 same-lane "
            "producer/consumer overlaps and zero cross-lane overlaps. The original cross-thread RAW "
            "label is rejected; the runtime-true mask and numeric mismatch remain separate observations."
        ),
    ),
    TritonFaultStrategy(
        suffix="m10-masked-load-roundtrip",
        rule_id="MUT-M10-GLOBAL-RAW-MASKED-LOAD",
        family="GLOBAL_MEMORY_ORDER",
        name="M10 runtime-true masked consumer",
        description=(
            "Add an output-neutral global round trip with a runtime all-true mask on the consumer load, "
            "forcing incompatible store/load layouts without changing source semantics."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    pipeguard_rows = tl.arange(0, 8)[:, None]
    pipeguard_cols = tl.arange(0, 256)[None, :]
    pipeguard_offsets = pipeguard_slot * 2048 + pipeguard_rows * 256 + pipeguard_cols
    pipeguard_mask = tl.load(pipeguard_wide + pipeguard_offsets) != 0
    pipeguard_aliases = pipeguard_scratch + pipeguard_offsets
    pipeguard_zero = tl.zeros((8, 256), tl.float32)
    tl.store(pipeguard_aliases, pipeguard_zero)
    pipeguard_reloaded = tl.load(pipeguard_aliases, mask=pipeguard_mask, other=0.0)
    return tl.max(tl.max(pipeguard_reloaded, axis=1), axis=0)
''',
        scratch_type="*fp32",
        auxiliary_type="*i8",
        adjudication_class="DATA_HAZARD",
        adjudication_status="CONFIRMED",
        adjudication_method="MANUAL_PROTOCOL_TRACE",
        adjudication_rationale=(
            "The runtime mask buffer is all true, so the masked load is semantically an ordinary reload. "
            "Eight of twenty workloads mismatch without a barrier and all twenty pass with a barrier."
        ),
    ),
    TritonFaultStrategy(
        suffix="m11-relaxed-atomic-roundtrip",
        rule_id="MUT-M11-RELAXED-ATOMIC-RAW",
        family="ATOMIC_MEMORY_ORDER",
        name="M11 relaxed atomic producer followed by load",
        description=(
            "Add an output-neutral relaxed GPU-scope atomic exchange followed by an ordinary load of the "
            "same unique addresses."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    pipeguard_rows = tl.arange(0, 8)[:, None]
    pipeguard_cols = tl.arange(0, 256)[None, :]
    pipeguard_offsets = pipeguard_slot * 2048 + pipeguard_rows * 256 + pipeguard_cols
    pipeguard_aliases = pipeguard_scratch + pipeguard_offsets
    pipeguard_zero = tl.zeros((8, 256), tl.int32)
    tl.atomic_xchg(pipeguard_aliases, pipeguard_zero, sem="relaxed", scope="gpu")
    pipeguard_reloaded = tl.load(pipeguard_aliases)
    return tl.max(tl.max(pipeguard_reloaded, axis=1), axis=0)
''',
        scratch_type="*i32",
        auxiliary_type="*fp16",
        adjudication_class="DATA_HAZARD",
        adjudication_status="PROVISIONAL",
        adjudication_method="MANUAL_PROTOCOL_TRACE",
        adjudication_rationale=(
            "Seven of twenty workloads mismatch and all twenty pass after a CTA barrier. The result is kept "
            "provisional because the precise Triton/PTX guarantee for relaxed atomic-to-non-atomic ordering "
            "must be stated before treating it as a strongest semantics-preserving positive."
        ),
    ),
    TritonFaultStrategy(
        suffix="m12-masked-other-constancy",
        rule_id="MUT-M12-MASKED-OTHER-CONSTANCY",
        family="COMPILER_ANALYSIS",
        name="M12 masked-load nonconstant other",
        description=(
            "Add an output-neutral masked-load canary whose nonconstant other operand must contribute to "
            "AxisInfo constancy."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    pipeguard_offsets = tl.arange(0, 256)
    pipeguard_groups = pipeguard_offsets // 16
    pipeguard_outer_mask = (pipeguard_offsets % 16) < 8
    pipeguard_varying_other = (pipeguard_offsets % 2) == 0
    pipeguard_loaded_mask = tl.load(
        pipeguard_scratch + pipeguard_slot * 16 + pipeguard_groups,
        mask=pipeguard_outer_mask,
        other=pipeguard_varying_other.to(tl.int32),
    )
    pipeguard_values = tl.load(
        pipeguard_wide + pipeguard_slot * 256 + pipeguard_offsets,
        mask=pipeguard_loaded_mask != 0,
        other=0,
    )
    return tl.sum(pipeguard_values, axis=0) - 192
''',
        scratch_type="*i32",
        auxiliary_type="*i32",
        adjudication_class="UNKNOWN",
        adjudication_status="CONFIRMED",
        adjudication_method="REFERENCE_ORACLE",
        adjudication_rationale=(
            "All twenty workloads mismatch on the pinned compiler. An equivalent explicit tl.where control "
            "passes all twenty. This is compiler wrong-code fixed upstream by commit 36cb2499e0 and is outside "
            "PipeGuard's synchronization scope."
        ),
    ),
    TritonFaultStrategy(
        suffix="m13-nondominating-assume",
        rule_id="MUT-M13-NONDOMINATING-ASSUME",
        family="CONTROL_FLOW_ANALYSIS",
        name="M13 non-dominating assume in flattened loop",
        description=(
            "Add an output-neutral zero-trip flattened-loop canary with an assumption in a mutually exclusive "
            "branch."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    pipeguard_control = pipeguard_scratch + pipeguard_slot * 66
    pipeguard_m = tl.load(pipeguard_control)
    pipeguard_n = tl.load(pipeguard_control + 1)
    pipeguard_input = pipeguard_control + 2
    pipeguard_sink = pipeguard_wide + pipeguard_slot * 2048
    pipeguard_offsets = tl.arange(0, 64)
    pipeguard_acc = tl.zeros((64,), dtype=tl.float32)
    if pipeguard_m == 0:
        tl.assume(pipeguard_n > 0)
        pipeguard_acc += tl.load(pipeguard_input + pipeguard_offsets)
    else:
        for pipeguard_i in tl.range(0, pipeguard_m, flatten=True):
            for pipeguard_j in range(0, pipeguard_n):
                pipeguard_acc += tl.load(pipeguard_input + pipeguard_j * 64 + pipeguard_offsets)
                tl.store(
                    pipeguard_sink + (pipeguard_i * 8 + pipeguard_j) * 64 + pipeguard_offsets,
                    pipeguard_acc,
                )
    return tl.max(tl.abs(pipeguard_acc), axis=0)
''',
        scratch_type="*i32",
        auxiliary_type="*fp32",
        adjudication_class="UNKNOWN",
        adjudication_status="CONFIRMED",
        adjudication_method="REFERENCE_ORACLE",
        adjudication_rationale=(
            "At runtime m=4 and n=0, so the loop must execute zero iterations and the canary must be zero. "
            "All twenty workloads mismatch with flatten=True; the flatten=False control passes all twenty. "
            "This compiler control-flow defect is outside PipeGuard's synchronization scope."
        ),
    ),
    TritonFaultStrategy(
        suffix="m14-cross-warp-roundtrip",
        rule_id="MUT-M14-GLOBAL-CROSS-WARP-RAW",
        family="GLOBAL_MEMORY_ORDER",
        name="M14 cross-warp global RAW round trip",
        description=(
            "Store a complete scratch tile and reload the identical addresses through a +64 mod 256 "
            "column permutation, creating an exact cross-warp RAW dependency without a CTA barrier."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    producer_rows = tl.arange(0, 8)[:, None]
    producer_cols = tl.arange(0, 256)[None, :]
    producer_offsets = pipeguard_slot * 2048 + producer_rows * 256 + producer_cols
    producer_aliases = pipeguard_scratch + producer_offsets
    tl.store(producer_aliases, tl.zeros((8, 256), tl.float32))
    consumer_rows = tl.arange(0, 8)[:, None]
    consumer_cols = (tl.arange(0, 256)[None, :] + 64) % 256
    consumer_offsets = pipeguard_slot * 2048 + consumer_rows * 256 + consumer_cols
    consumer_aliases = pipeguard_scratch + consumer_offsets
    reloaded = tl.load(consumer_aliases)
    return tl.max(tl.max(reloaded, axis=1), axis=0)
''',
        scratch_type="*fp32",
        auxiliary_type="*fp16",
        adjudication_class="DATA_HAZARD",
        adjudication_status="CONFIRMED",
        adjudication_method="MANUAL_PROTOCOL_TRACE",
        adjudication_rationale=(
            "The source permutation is bijective. Exact final-lowering analysis finds 2,048 overlapping "
            "byte regions, all crossing physical warps, with no barrier or typed visibility operation."
        ),
        kernel_ids=("triton-dense-gemm",),
    ),
    TritonFaultStrategy(
        suffix="m15-cross-warp-barrier-control",
        rule_id="MUT-M15-GLOBAL-CROSS-WARP-BARRIER-CONTROL",
        family="GLOBAL_MEMORY_ORDER",
        name="M15 synchronized cross-warp global RAW control",
        description=(
            "Use the exact M14 address permutation with one CTA barrier between the producer store and "
            "consumer load."
        ),
        helper='''@triton.jit
def _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pipeguard_slot):
    producer_rows = tl.arange(0, 8)[:, None]
    producer_cols = tl.arange(0, 256)[None, :]
    producer_offsets = pipeguard_slot * 2048 + producer_rows * 256 + producer_cols
    producer_aliases = pipeguard_scratch + producer_offsets
    tl.store(producer_aliases, tl.zeros((8, 256), tl.float32))
    tl.debug_barrier()
    consumer_rows = tl.arange(0, 8)[:, None]
    consumer_cols = (tl.arange(0, 256)[None, :] + 64) % 256
    consumer_offsets = pipeguard_slot * 2048 + consumer_rows * 256 + consumer_cols
    consumer_aliases = pipeguard_scratch + consumer_offsets
    reloaded = tl.load(consumer_aliases)
    return tl.max(tl.max(reloaded, axis=1), axis=0)
''',
        scratch_type="*fp32",
        auxiliary_type="*fp16",
        adjudication_class="CORRECT",
        adjudication_status="CONFIRMED",
        adjudication_method="MANUAL_PROTOCOL_TRACE",
        adjudication_rationale=(
            "The exact M14 lane-byte map is retained, while TTGIR, PTX, SASS, and normalized events all "
            "place a full-CTA barrier between the producer and consumer."
        ),
        kernel_ids=("triton-dense-gemm",),
    ),
)

STRATEGIES = {item.suffix: item for item in _STRATEGIES}
RULES = tuple({
    "rule_id": item.rule_id,
    "family": item.family,
    "name": item.name,
    "description": item.description,
    "computational_intent": "PRESERVE_OUTPUT",
} for item in _STRATEGIES)


def strategy_from_variant(variant_id: str) -> TritonFaultStrategy | None:
    return next((item for item in _STRATEGIES if variant_id.endswith(f"-source-{item.suffix}")), None)


def _replace_once(text: str, old: str, new: str, label: str) -> str:
    count = text.count(old)
    if count != 1:
        raise RuntimeError(f"{label}: expected one occurrence, found {count}: {old!r}")
    return text.replace(old, new, 1)


def mutate_source(source: str, kernel_id: str, suffix: str) -> str:
    strategy = STRATEGIES[suffix]
    label = f"{kernel_id}:{suffix}"
    marker = "import triton.language as tl\n"
    source = _replace_once(source, marker, marker + "\n\n" + strategy.helper + "\n", label)
    if kernel_id == "triton-dense-gemm":
        source = _replace_once(
            source,
            "        a_ptr, b_ptr, c_ptr,\n",
            "        a_ptr, b_ptr, c_ptr, pipeguard_scratch, pipeguard_wide,\n",
            label,
        )
        source = _replace_once(
            source,
            "    c = accumulator.to(tl.float16)\n",
            "    pipeguard_fault = _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, pid)\n"
            "    c = (accumulator + pipeguard_fault).to(tl.float16)\n",
            label,
        )
    elif kernel_id == "triton-grouped-gemm":
        source = _replace_once(
            source,
            "    group_c_ptrs,\n",
            "    group_c_ptrs,\n    pipeguard_scratch,\n    pipeguard_wide,\n",
            label,
        )
        source = _replace_once(
            source,
            "            c = accumulator.to(tl.float16)\n",
            "            pipeguard_fault = _pipeguard_global_roundtrip_fault(\n"
            "                pipeguard_scratch, pipeguard_wide, tile_idx\n"
            "            )\n"
            "            c = (accumulator + pipeguard_fault).to(tl.float16)\n",
            label,
        )
    elif kernel_id == "triton-fused-attention":
        source = _replace_once(
            source,
            "              Z, H, desc_q, desc_k, desc_v, desc_o, N_CTX,  #\n",
            "              Z, H, desc_q, desc_k, desc_v, desc_o, pipeguard_scratch, pipeguard_wide, N_CTX,  #\n",
            label,
        )
        source = _replace_once(
            source,
            "    acc = acc / l_i[:, None]\n",
            "    acc = acc / l_i[:, None]\n"
            "    pipeguard_slot = off_hz * tl.cdiv(N_CTX, BLOCK_M) + start_m\n"
            "    pipeguard_fault = _pipeguard_global_roundtrip_fault(\n"
            "        pipeguard_scratch, pipeguard_wide, pipeguard_slot\n"
            "    )\n"
            "    acc += pipeguard_fault\n",
            label,
        )
    elif kernel_id == "triton-layer-norm":
        source = _replace_once(
            source,
            "    Y,  # pointer to the output\n",
            "    Y,  # pointer to the output\n    pipeguard_scratch,\n    pipeguard_wide,\n",
            label,
        )
        source = _replace_once(
            source,
            "    # Normalize and apply linear transformation\n",
            "    pipeguard_fault = _pipeguard_global_roundtrip_fault(pipeguard_scratch, pipeguard_wide, row)\n"
            "    # Normalize and apply linear transformation\n",
            label,
        )
        source = _replace_once(
            source,
            "        y = x_hat * w + b\n",
            "        y = x_hat * w + b + pipeguard_fault\n",
            label,
        )
    elif kernel_id == "triton-softmax":
        source = _replace_once(
            source,
            "def softmax_kernel(output_ptr, input_ptr, input_row_stride",
            "def softmax_kernel(output_ptr, input_ptr, pipeguard_scratch, pipeguard_wide, input_row_stride",
            label,
        )
        source = _replace_once(
            source,
            "        softmax_output = numerator / denominator\n",
            "        softmax_output = numerator / denominator\n"
            "        pipeguard_fault = _pipeguard_global_roundtrip_fault(\n"
            "            pipeguard_scratch, pipeguard_wide, row_idx\n"
            "        )\n"
            "        softmax_output += pipeguard_fault\n",
            label,
        )
    else:
        raise ValueError(f"unsupported Triton fault kernel: {kernel_id}")
    return source


_INSERT_AFTER = {
    "triton-dense-gemm": ("c_ptr", 3),
    "triton-grouped-gemm": ("group_c_ptrs", 3),
    "triton-fused-attention": ("desc_o", 8),
    "triton-layer-norm": ("Y", 2),
    "triton-softmax": ("input_ptr", 2),
}


def augment_triton_recipe(recipe: dict[str, Any], variant_id: str) -> dict[str, Any]:
    strategy = strategy_from_variant(variant_id)
    if strategy is None:
        return recipe
    result = copy.deepcopy(recipe)
    compiler = result["compiler"]
    signature = compiler["signature"]
    anchor, position = _INSERT_AFTER[result["kernel_id"]]
    if anchor not in signature:
        raise ValueError(f"missing Triton signature anchor: {anchor}")
    entries = list(signature.items())
    entries.extend([
        ("pipeguard_scratch", strategy.scratch_type),
        ("pipeguard_wide", strategy.auxiliary_type),
    ])
    compiler["signature"] = dict(entries)
    evidence = compiler.get("binding_evidence")
    if evidence is not None:
        old = evidence.get("pointer_allocations", {})
        shifted = {
            str(int(index) + (2 if int(index) >= position else 0)): allocation
            for index, allocation in old.items()
        }
        shifted[str(position)] = "global:pipeguard_scratch"
        shifted[str(position + 1)] = "global:pipeguard_auxiliary"
        evidence["pointer_allocations"] = dict(sorted(shifted.items(), key=lambda item: int(item[0])))
        if strategy.suffix in {
            "m9-masked-store-roundtrip",
            "m10-masked-load-roundtrip",
        }:
            evidence["allocation_values"] = {
                "global:pipeguard_auxiliary": {
                    "kind": "CONSTANT",
                    "value": 1,
                },
            }
    return result


def allocate_fault_buffers(torch: Any, variant_id: str, slots: int) -> tuple[Any, Any] | None:
    strategy = strategy_from_variant(variant_id)
    if strategy is None:
        return None
    slots = max(int(slots), 1)
    if strategy.suffix in {
        "m8-global-roundtrip",
        "m14-cross-warp-roundtrip",
        "m15-cross-warp-barrier-control",
    }:
        return (
            torch.full((slots, 8, 256), 8192.0, device="cuda", dtype=torch.float32),
            torch.randn((slots, 8, 256), device="cuda", dtype=torch.float16),
        )
    if strategy.suffix in {"m9-masked-store-roundtrip", "m10-masked-load-roundtrip"}:
        return (
            torch.full((slots, 8, 256), 8192.0, device="cuda", dtype=torch.float32),
            torch.ones((slots, 8, 256), device="cuda", dtype=torch.int8),
        )
    if strategy.suffix == "m11-relaxed-atomic-roundtrip":
        return (
            torch.full((slots, 8, 256), 8192, device="cuda", dtype=torch.int32),
            torch.randn((slots, 8, 256), device="cuda", dtype=torch.float16),
        )
    if strategy.suffix == "m12-masked-other-constancy":
        return (
            torch.ones((slots, 16), device="cuda", dtype=torch.int32),
            torch.ones((slots, 256), device="cuda", dtype=torch.int32),
        )
    if strategy.suffix == "m13-nondominating-assume":
        scratch = torch.zeros((slots, 66), device="cuda", dtype=torch.int32)
        scratch[:, 0] = 4
        scratch[:, 1] = 0
        scratch[:, 2:] = (torch.arange(64, device="cuda", dtype=torch.int32) % 7) + 1
        return scratch, torch.full((slots, 2048), -7777.0, device="cuda", dtype=torch.float32)
    raise AssertionError(strategy.suffix)
