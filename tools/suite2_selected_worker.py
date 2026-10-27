from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src"), str(ROOT / "tools"), str(ROOT / "tests")]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_once(path: Path, data: bytes) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    with os.fdopen(fd, "wb") as stream:
        stream.write(data)
    return {"path": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


_TMA_STORE_WAIT_OP = "ttng.async_tma_store_wait"
# MLIR custom-assembly attribute dictionary printed after the op name, e.g.
# `ttng.async_tma_store_wait {pendings = 0 : i32, read_only} loc(#loc8)`.
_TMA_STORE_WAIT_TEXT = re.compile(r"ttng\.async_tma_store_wait\s*(?:\{([^{}]*)\})?")
_TMA_STORE_WAIT_PENDINGS = re.compile(r"^pendings\s*=\s*(-?\d+)\s*:")


def tma_store_wait_unit_attributes(raw_path: Path) -> dict[int, list[str]]:
    """Faithfully extract unit attributes of every ttng.async_tma_store_wait.

    The Triton python bindings cannot read unit attributes, so triton_capture
    demands binding evidence for them. Fabricating empty evidence would
    silently upgrade a `{read_only}` wait to the stronger GLOBAL wait mode
    (destination writes modelled as complete). This helper recovers the real
    attributes from the textual TTGIR and raises CaptureError (fail-closed ->
    capture_status UNSUPPORTED) whenever the attributes cannot be established
    unambiguously; it never defaults to GLOBAL.
    """
    import triton_capture
    from triton._C.libtriton import ir, nvidia

    def unresolved(reason: str, **details: Any) -> Exception:
        return triton_capture.CaptureError(
            "TRITON_TMA_STORE_WAIT_EVIDENCE_UNRESOLVED",
            f"TMA store wait unit attributes could not be established: {reason}",
            {"path": str(raw_path), **details},
        )

    context = ir.context()
    ir.load_dialects(context)
    nvidia.load_dialects(context)
    try:
        module = ir.parse_mlir_module(str(raw_path), context)
        module.verify()
    except (OSError, RuntimeError) as error:
        raise unresolved("TTGIR module could not be parsed", error=str(error))
    operations: list[Any] = []
    module.walk(operations.append)
    wait_indices = [
        index
        for index, operation in enumerate(operations)
        if operation.get_name() == _TMA_STORE_WAIT_OP
    ]

    raw_text = raw_path.read_text()
    textual: list[tuple[int, list[str]]] = []
    for match in _TMA_STORE_WAIT_TEXT.finditer(raw_text):
        body = match.group(1)
        if body is None:
            raise unresolved(
                "textual occurrence has no attribute dictionary",
                offset=match.start(),
            )
        pendings = None
        units: list[str] = []
        for entry in body.split(","):
            entry = entry.strip()
            if not entry:
                raise unresolved(
                    "empty attribute entry in attribute dictionary",
                    offset=match.start(),
                )
            pending_match = _TMA_STORE_WAIT_PENDINGS.match(entry)
            if pending_match:
                pendings = int(pending_match.group(1))
            elif "=" in entry:
                raise unresolved(
                    "unrecognised non-unit attribute on TMA store wait",
                    attribute=entry,
                    offset=match.start(),
                )
            else:
                units.append(entry)
        if pendings is None:
            raise unresolved(
                "textual occurrence lacks a static pendings attribute",
                offset=match.start(),
            )
        textual.append((pendings, sorted(units)))

    if len(textual) != len(wait_indices):
        raise unresolved(
            "textual and walk-order occurrence counts disagree",
            textual_occurrences=len(textual),
            walk_occurrences=len(wait_indices),
        )
    # async_tma_store_wait has no regions, so walk order and textual order
    # coincide; the per-occurrence pendings cross-check below rejects any
    # pairing in which that invariant does not hold.
    evidence: dict[int, list[str]] = {}
    for index, (pendings, units) in zip(wait_indices, textual):
        walk_pendings = operations[index].get_int_attr("pendings")
        if walk_pendings is None or int(walk_pendings) != pendings:
            raise unresolved(
                "pendings cross-check between walk order and text failed",
                operation_index=index,
                walk_pendings=walk_pendings,
                textual_pendings=pendings,
            )
        evidence[index] = units
    return evidence


def triton_capture_with_evidence(
    raw_path: Path,
    evidence_path: Path,
    family: str,
    label: str,
    *,
    descriptor_shape: list[int] | None = None,
    final_ptx_path: Path | None = None,
) -> dict[str, Any]:
    import triton_capture

    pointer_allocations = {}
    if family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF", "MEMORY", "CTA-BARRIER", "MBARRIER-DIVERSE"}:
        pointer_allocations = {"0": "global:data", "1": "global:output"}
    elif family in {"CP-WAIT", "CP-OVERLAP", "CP-ASYNC"}:
        pointer_allocations = {"0": "global:input", "1": "global:output"}
    elif family in {"TMA-OBSERVE", "TMA-PROXY", "TMA-LOAD", "TMA-SYNC"}:
        pointer_allocations = {"5": "global:input", "6": "global:output"}
    elif family == "TMA-STORE":
        pointer_allocations = {"5": "global:output", "6": "global:sink"}
    elif family == "PROXY-STORE":
        pointer_allocations = {
            "5": "global:output",
            "6": "global:sink",
        }
    evidence = {
        "evidence_version": "triton-binding-evidence-v1",
        "source_ir_sha256": sha(raw_path),
        "unit_attributes": {},
        "array_attributes": {},
        "pointer_allocations": pointer_allocations,
        "scalar_values": {},
    }
    tma_store_wait_evidence: dict[int, list[str]] | None = None
    for _attempt in range(64):
        evidence_path.write_text(json.dumps(evidence, indent=2, sort_keys=True))
        try:
            return triton_capture.capture_file(
                raw_path,
                source=f"suite2:{label}",
                command="suite2-selected210-triton-compile-only",
                grid_ctas=1,
                binding_evidence_path=evidence_path,
                final_ptx_path=final_ptx_path,
            )
        except triton_capture.CaptureError as error:
            operation_index = error.details.get("operation_index")
            if not isinstance(operation_index, int):
                raise
            if error.code == "TRITON_BINDING_EVIDENCE_REQUIRED":
                if "warp_specialize" in str(error):
                    evidence["array_attributes"][str(operation_index)] = {
                        "partition_num_warps": [4],
                        "warp_group_start_ids": [4],
                    }
                elif "TMA store wait" in str(error):
                    # Supply the faithfully extracted unit attributes; a
                    # read_only wait must never degrade into the stronger
                    # GLOBAL default. Extraction failures raise CaptureError
                    # (fail-closed -> UNSUPPORTED) instead of guessing.
                    if tma_store_wait_evidence is None:
                        tma_store_wait_evidence = tma_store_wait_unit_attributes(raw_path)
                    if operation_index not in tma_store_wait_evidence:
                        raise triton_capture.CaptureError(
                            "TRITON_TMA_STORE_WAIT_EVIDENCE_UNRESOLVED",
                            "capture demanded TMA store wait evidence for an "
                            "operation the extractor did not identify",
                            {"operation_index": operation_index},
                        ) from error
                    evidence["unit_attributes"][str(operation_index)] = (
                        tma_store_wait_evidence[operation_index]
                    )
                else:
                    # Fail closed: never fabricate binding evidence for an
                    # unrecognised demand.
                    raise
                continue
            if error.code != "TRITON_TMA_DESCRIPTOR_UNRESOLVED":
                raise
            is_named_store = family == "TMA-STORE"
            is_proxy_store = family == "PROXY-STORE"
            allocation = (
                "global:output"
                if family in {"TMA-STORE", "TMA-STORE-ENDPOINT", "PROXY-STORE"}
                else "global:input"
            )
            shape = descriptor_shape or ([64, 32] if is_named_store else [32, 32])
            evidence["array_attributes"][str(operation_index)] = {
                "descriptor": {
                    "allocation_id": allocation,
                    "coordinate_order": [0, 1],
                    "element_bytes": 2,
                    "shape": shape,
                    "strides": [shape[1], 1],
                    "tile_shape": [64, 64] if is_proxy_store else [32, 32],
                }
            }
    raise AssertionError("descriptor evidence retry did not return or raise")


def load_source(spec: dict[str, Any]) -> Any:
    path = ROOT / spec["source"]["path"]
    module_spec = importlib.util.spec_from_file_location("suite2_exec_" + spec["case_id"].replace("-", "_"), path)
    if module_spec is None or module_spec.loader is None:
        raise ImportError(path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def tilelang_pass_config(family: str) -> dict[str, bool]:
    if family in {"CTA-HANDOFF", "CTA-BARRIER"}:
        return {"tl.disable_thread_storage_sync": True}
    return {}


def wgmma_numeric_observation(value: Any, repeats: int) -> dict[str, Any]:
    values = value if isinstance(value, (list, tuple)) else [value]
    tensor = next((item for item in values if hasattr(item, "detach")), None)
    if tensor is None:
        return {"status": "NOT_APPLICABLE"}
    observed = tensor.detach().float()
    expected = float(16 * repeats)
    finite = bool(observed.isfinite().all().item())
    max_abs_error = float((observed - expected).abs().max().item())
    return {
        "status": "MATCHED_REFERENCE" if finite and max_abs_error <= 1e-3 else "MISMATCHED_REFERENCE",
        "reference": "ones(64x16) @ ones(16x64) accumulated wgmma_repeats times",
        "expected_value": expected,
        "shape": list(tensor.shape),
        "minimum": float(observed.min().item()),
        "maximum": float(observed.max().item()),
        "max_abs_error": max_abs_error,
        "finite": finite,
        "safety_oracle": False,
    }


def proxy_numeric_observation(value: Any, repeats: int, result_class: str) -> dict[str, Any]:
    values = value if isinstance(value, (list, tuple)) else [value]
    tensor = next((item for item in values if hasattr(item, "detach")), None)
    if tensor is None:
        return {"status": "NOT_APPLICABLE"}
    observed = tensor.detach().float()
    expected = float(16 * repeats if result_class == "SYNC_HAZARD" else repeats)
    finite = bool(observed.isfinite().all().item())
    error = float((observed - expected).abs().max().item())
    return {
        "status": "MATCHED_REFERENCE" if finite and error <= 1e-3 else "MISMATCHED_REFERENCE",
        "expected_value": expected,
        "shape": list(tensor.shape),
        "minimum": float(observed.min().item()),
        "maximum": float(observed.max().item()),
        "max_abs_error": error,
        "finite": finite,
        "safety_oracle": False,
    }


def tensor_observation(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {"status": "NOT_APPLICABLE"}
    values = value if isinstance(value, (list, tuple)) else [value]
    samples = []
    for item in values:
        if not hasattr(item, "detach"):
            continue
        flattened = item.detach().reshape(-1)
        samples.append({
            "label": label,
            "shape": list(item.shape),
            "values": flattened[:8].cpu().tolist(),
        })
    return {
        "status": "OBSERVED_NOT_SAFETY_ORACLE" if samples else "NOT_APPLICABLE",
        "samples": samples,
    }


def tilelang_run(spec: dict[str, Any], module: Any, output: Path) -> dict[str, Any]:
    import torch
    import tilelang
    from tilelang import tvm

    family = spec["template_family"]
    function = (
        module.pipeguard_resolve()
        if hasattr(module, "pipeguard_resolve")
        else module.kernel
    )
    if spec["primitive"] == "Proxy":
        if spec["result_class"] == "SYNC_HAZARD":
            args = [
                torch.ones((64, 16), device="cuda", dtype=torch.float16),
                torch.ones((16, 64), device="cuda", dtype=torch.float16),
            ]
            out_idx = 2
        else:
            args = []
            out_idx = 0
    elif family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        size = max(16, spec["semantic_knobs"]["elements"] + 1)
        args = [torch.zeros((size,), device="cuda", dtype=torch.int32)]
        out_idx = 1
    elif family == "CTA-BARRIER":
        args = []
        out_idx = 0
    elif family == "MBARRIER-DIVERSE":
        args = [torch.zeros((256,), device="cuda", dtype=torch.int32)]
        out_idx = 1
    elif family == "MBARRIER-DIVERSE":
        args = [
            torch.zeros((256,), device="cuda", dtype=torch.int32),
            torch.zeros((32,), device="cuda", dtype=torch.int32),
        ]
    elif family in {"MB-REUSE", "MBARRIER"}:
        args = []
        out_idx = []
    elif family in {"CP-WAIT", "CP-OVERLAP", "CP-ASYNC"}:
        byte_count = (4, 8, 16)[spec["variant"] % 3]
        args = [torch.zeros((max(32, 2 * byte_count),), device="cuda", dtype=torch.uint8)]
        out_idx = 1
    elif family in {"TMA-OBSERVE", "TMA-PROXY"}:
        width = 16 * (spec["variant"] + 1)
        args = [torch.zeros((width,), device="cuda", dtype=torch.uint8)]
        out_idx = 1
    elif family == "TMA-LOAD":
        args = [torch.zeros((32,), device="cuda", dtype=torch.uint8)]
        out_idx = 1
    elif family == "TMA-SYNC":
        width = 16 * (spec["variant"] + 1)
        args = [torch.zeros((width,), device="cuda", dtype=torch.uint8)]
        out_idx = 1
    elif family in {"WGMMA-ENDPOINT", "WGMMA"}:
        args = [
            torch.ones((64, 16), device="cuda", dtype=torch.float16),
            torch.ones((16, 64), device="cuda", dtype=torch.float16),
        ]
        out_idx = 2
    elif family in {"TMA-STORE-ENDPOINT", "TMA-STORE"}:
        args = []
        out_idx = [0, 1]
    else:
        raise KeyError(family)

    pass_config = tilelang_pass_config(family)
    if spec.get("standalone_source", False):
        pass_config = {"tl.disable_thread_storage_sync": True}
    compiler_overrides = []
    if spec["primitive"] == "Proxy" and spec["result_class"] == "DATA_HAZARD":
        class _IdentityPass:
            def __call__(self, mod):
                return mod
        tilelang.cuda.transform.InjectFenceProxy = lambda: _IdentityPass()
        compiler_overrides.append("tilelang.InjectFenceProxy=identity")
    with tvm.transform.PassContext(opt_level=3, config=pass_config):
        compiled = tilelang.compile(
            function,
            out_idx=out_idx,
            target="cuda",
            pass_configs=pass_config,
        )
    kernel_source = compiled.get_kernel_source()
    source_ref = write_once(output / "kernel.cu", kernel_source.encode())
    library = output / "kernel.so"
    compiled.export_library(str(library))
    library_ref = {"path": library.name, "sha256": sha(library), "bytes": library.stat().st_size}
    numeric = {"status": "NOT_RUN_COMPILE_CAPTURE_ONLY"}

    target = tvm.target.Target({"kind": "cuda", "arch": "sm_90a", "thread_warp_size": 32, "max_num_threads": 1024})
    with target, tvm.transform.PassContext(opt_level=3, config=pass_config):
        raw = tilelang.lower(function, target=target, enable_host_codegen=False, enable_device_compile=False).device_mod
    raw_path = output / "raw.tvm.json"
    write_once(raw_path, tvm.ir.save_json(raw).encode())
    descriptor_evidence_path = None
    descriptor_evidence_ref = None
    if spec["primitive"] == "WGMMA" or (
        spec["primitive"] == "Proxy" and spec["result_class"] == "SYNC_HAZARD"
    ):
        descriptor_evidence_path = output / "descriptor-evidence.json"
        descriptor_evidence = {
            "evidence_version": "tilelang-tma-descriptor-evidence-v1",
            "source_ir_sha256": sha(raw_path),
            "descriptors": {
                "A_desc": {
                    "allocation_id": "global:A",
                    "element_bytes": 2,
                    "shape": [64, 16],
                    "strides": [16, 1],
                    "tile_shape": [64, 16],
                    "coordinate_order": [0, 1],
                },
                "B_desc": {
                    "allocation_id": "global:B",
                    "element_bytes": 2,
                    "shape": [16, 64],
                    "strides": [64, 1],
                    "tile_shape": [16, 64],
                    "coordinate_order": [0, 1],
                },
            },
        }
        descriptor_evidence_ref = write_once(
            descriptor_evidence_path,
            json.dumps(descriptor_evidence, indent=2, sort_keys=True).encode(),
        )
    import tilelang_capture
    snapshot = tilelang_capture.capture_file(
        raw_path,
        source=f"suite2-executable:{spec['case_id']}",
        command="suite2-executable-tilelang",
        descriptor_evidence_path=descriptor_evidence_path,
        launch_grid=[1, 1, 1],
        selected_cta=(0, 0, 0),
    )
    snapshot_ref = write_once(output / "snapshot.json", json.dumps(snapshot, indent=2, sort_keys=True).encode())
    return {
        "execution_status": "COMPILED", "capture_status": "CAPTURED",
        "executable": library_ref, "device_source": source_ref,
        "raw": {"path": raw_path.name, "sha256": sha(raw_path)}, "snapshot": snapshot_ref,
        "descriptor_evidence": descriptor_evidence_ref,
        "compiler_overrides": compiler_overrides,
        "numeric": numeric,
    }


def triton_num_warps(threads: int) -> int:
    requested = max(1, (threads + 31) // 32)
    return 1 << (requested - 1).bit_length()


def triton_prepare(spec: dict[str, Any], module: Any, torch: Any):
    from triton.experimental import gluon
    from triton.experimental.gluon import language as ttgl

    family = spec["template_family"]
    buggy = spec["polarity"] == "BUG"
    threads = spec["semantic_knobs"]["threads"]
    num_warps = triton_num_warps(threads)
    kernel = (
        module.pipeguard_resolve()
        if hasattr(module, "pipeguard_resolve")
        else module.kernel
    )
    kwargs = {"num_warps": num_warps, "num_ctas": 1}
    if (
        not spec.get("standalone_source", False)
        and spec["primitive"] != "cp.async"
    ):
        kwargs["BUGGY"] = buggy
    if family == "MB-REUSE":
        kwargs["DEPTH"] = spec["variant"] + 2
    if spec["primitive"] == "Proxy":
        kwargs["num_warps"] = 4
        if spec["result_class"] == "SYNC_HAZARD":
            args = [torch.empty((64, 64), device="cuda", dtype=torch.float32)]
            kwargs["OPERAND_CASE"] = False
        else:
            out = torch.empty((64, 64), device="cuda", dtype=torch.float16)
            sink = torch.empty_like(out)
            layout = ttgl.NVMMASharedLayout(128, 16, rank=2, cga_layout=())
            desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(out, [64, 64], layout)
            args = [desc, out, sink]
    elif family in {"MEM-RAW", "MEM-WAW", "CTA-HANDOFF"}:
        args = [
            torch.zeros((max(16, spec["semantic_knobs"]["elements"] + 1),), device="cuda", dtype=torch.int32),
            torch.zeros((threads,), device="cuda", dtype=torch.int32),
        ]
    elif family == "CTA-BARRIER":
        args = [torch.zeros((128,), device="cuda", dtype=torch.int32), torch.zeros((128,), device="cuda", dtype=torch.int32)]
    elif family == "MBARRIER-DIVERSE":
        args = [
            torch.zeros((256,), device="cuda", dtype=torch.int32),
            torch.zeros((32,), device="cuda", dtype=torch.int32),
        ]
    elif family in {"MB-REUSE", "MBARRIER"}:
        args = []
    elif spec["primitive"] == "cp.async":
        args = [
            torch.zeros((16 * threads,), device="cuda", dtype=torch.int32),
            torch.zeros((threads,), device="cuda", dtype=torch.int32),
        ]
    elif family in {"CP-WAIT", "CP-OVERLAP", "CP-ASYNC"}:
        args = [torch.zeros((2 * threads,), device="cuda", dtype=torch.int32)]
    elif family in {"TMA-OBSERVE", "TMA-PROXY", "TMA-LOAD", "TMA-SYNC"}:
        block = 32
        transfers = spec["variant"] + 1 if family in {"TMA-OBSERVE", "TMA-PROXY", "TMA-SYNC"} else 1
        tensor = torch.zeros((transfers * block, block), device="cuda", dtype=torch.float16)
        out = torch.zeros((block, block), device="cuda", dtype=torch.float16)
        layout = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=())
        desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(tensor, [block, block], layout)
        args = [desc, tensor, out]
        kwargs["num_warps"] = 4
    elif family in {"WGMMA-ENDPOINT", "WGMMA"}:
        args = [torch.empty((64, 64), device="cuda", dtype=torch.float32)]
        kwargs["num_warps"] = 4
        kwargs["OPERAND_CASE"] = spec["result_class"] == "DATA_HAZARD"
    elif family in {"TMA-STORE-ENDPOINT", "TMA-STORE"}:
        block = 32
        out = torch.zeros((2 * block, block), device="cuda", dtype=torch.float16)
        sink = torch.zeros((1,), device="cuda", dtype=torch.float16)
        layout = ttgl.NVMMASharedLayout(swizzle_byte_width=32, element_bitwidth=16, rank=2, cga_layout=())
        desc = gluon.nvidia.hopper.TensorDescriptor.from_tensor(out, [block, block], layout)
        args = [desc] if family == "TMA-STORE-ENDPOINT" else [desc, out, sink]
        kwargs["num_warps"] = 4
        if family == "TMA-STORE-ENDPOINT":
            kwargs["SOURCE_CASE"] = spec["variant"] % 2 == 0
    else:
        raise KeyError(family)
    return kernel, args, kwargs


def triton_run(spec: dict[str, Any], module: Any, output: Path) -> dict[str, Any]:
    import torch
    family = spec["template_family"]
    kernel, args, kwargs = triton_prepare(spec, module, torch)
    compiler_overrides = []
    if spec["primitive"] == "Proxy" and spec["result_class"] == "DATA_HAZARD":
        from triton.backends.nvidia import compiler as triton_compiler
        triton_compiler.nvidia.passes.ttnvgpuir.add_proxy_fence_insertion = (
            lambda pm, capability: None
        )
        compiler_overrides.append("triton.add_proxy_fence_insertion=noop")
    compiled = kernel.warmup(*args, grid=(1,), **kwargs)
    asm = compiled.asm
    raw = str(asm["ttgir"])
    raw_ref = write_once(output / "raw.ttgir.mlir", raw.encode())
    refs = {}
    for key, suffix in (("ptx", "ptx"), ("cubin", "cubin")):
        value = asm.get(key)
        if value is not None:
            data = value if isinstance(value, bytes) else str(value).encode()
            refs[key] = write_once(output / f"kernel.{suffix}", data)
    numeric = {"status": "NOT_RUN_COMPILE_CAPTURE_ONLY"}
    descriptor_shape = None
    capture_family = spec["template_family"]
    if spec["primitive"] == "Proxy" and spec["result_class"] != "SYNC_HAZARD":
        descriptor_shape = [64, 64]
        capture_family = "PROXY-STORE"
    elif family in {"TMA-OBSERVE", "TMA-PROXY", "TMA-LOAD", "TMA-SYNC"}:
        transfers = spec["variant"] + 1 if family in {"TMA-OBSERVE", "TMA-PROXY", "TMA-SYNC"} else 1
        descriptor_shape = [transfers * 32, 32]
    snapshot = triton_capture_with_evidence(
        output / "raw.ttgir.mlir",
        output / "binding-evidence.json",
        capture_family,
        spec["case_id"],
        descriptor_shape=descriptor_shape,
        final_ptx_path=(
            output / "kernel.ptx"
            if "ptx" in refs
            and (
                spec["primitive"] == "WGMMA"
                or (
                    spec["primitive"] == "Proxy"
                    and spec["result_class"] == "SYNC_HAZARD"
                )
            )
            else None
        ),
    )
    snapshot_ref = write_once(output / "snapshot.json", json.dumps(snapshot, indent=2, sort_keys=True).encode())
    return {
        "execution_status": "COMPILED", "capture_status": "CAPTURED",
        "executable": refs.get("cubin"), "device_source": refs.get("ptx"),
        "raw": raw_ref, "snapshot": snapshot_ref, "numeric": numeric,
        "launch": {"num_warps": kwargs["num_warps"], "num_ctas": 1, "grid": [1]},
        "compiler_overrides": compiler_overrides,
        "tma_descriptor_shape": descriptor_shape,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.spec.read_text())
    module = load_source(spec)
    started = time.monotonic()
    try:
        result = tilelang_run(spec, module, args.output) if spec["dsl"] == "tilelang" else triton_run(spec, module, args.output)
        result["elapsed_seconds"] = time.monotonic() - started
    except Exception as error:
        capture_error = type(error).__name__ == "CaptureError" and hasattr(error, "diagnostic")
        result = {
            "execution_status": "COMPILED" if capture_error else "ERROR",
            "capture_status": "UNSUPPORTED" if capture_error else "ERROR",
            "diagnostic": error.diagnostic() if capture_error else type(error).__name__,
            "message": str(error),
            "traceback": traceback.format_exc(limit=30),
            "elapsed_seconds": time.monotonic() - started,
        }
    print(json.dumps(result, sort_keys=True))
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
