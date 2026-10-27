from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))

from mutation_suite_triton_faults import augment_triton_recipe  # noqa: E402
from upstream_common import atomic_write, pretty_json  # noqa: E402
from upstream_worker import _lower_tilelang, _lower_triton, _normalize_triton_ir  # noqa: E402


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _row_major_strides(shape: list[int]) -> list[int]:
    strides = [1] * len(shape)
    for index in range(len(shape) - 2, -1, -1):
        strides[index] = strides[index + 1] * shape[index + 1]
    return strides


def _tilelang_dtype_bytes(dtype: str) -> int:
    widths = {"float16": 2, "bfloat16": 2, "float32": 4}
    try:
        return widths[dtype]
    except KeyError as error:
        raise ValueError(f"unsupported Suite 4 TileLang descriptor dtype: {dtype}") from error


def _descriptor(
    allocation: str,
    shape: list[int],
    tile_shape: list[int],
    element_bytes: int,
) -> dict[str, Any]:
    return {
        "allocation_id": f"global:{allocation}",
        "element_bytes": element_bytes,
        "shape": shape,
        "strides": _row_major_strides(shape),
        "tile_shape": tile_shape,
        "coordinate_order": list(reversed(range(len(shape)))),
    }


def _tilelang_descriptor_evidence(
    case_id: str,
    recipe: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    kernel_id = case_id.split("-source-", 1)[0]
    materialization = recipe["materialization"]
    args = materialization.get("args", [])
    kwargs = materialization.get("kwargs", {})
    if kernel_id == "tilelang-dense-gemm":
        m, n, k = (int(kwargs[name]) for name in ("M", "N", "K"))
        block_m, block_n, block_k = (
            int(kwargs[name]) for name in ("block_M", "block_N", "block_K")
        )
        element_bytes = _tilelang_dtype_bytes(str(kwargs.get("dtype", "float16")))
        return {
            "A_desc": _descriptor("A", [m, k], [block_m, block_k], element_bytes),
            "B_desc": _descriptor("B", [k, n], [block_k, block_n], element_bytes),
        }
    if kernel_id == "tilelang-fused-attention":
        (
            batch,
            heads,
            seq_q,
            seq_kv,
            dim,
            _is_causal,
            block_m,
            block_n,
            _num_stages,
            _threads,
        ) = args
        element_bytes = 2
        q_shape = [int(batch), int(heads), int(seq_q), int(dim)]
        kv_shape = [int(batch), int(heads), int(seq_kv), int(dim)]
        return {
            "Q_desc": _descriptor(
                "Q", q_shape, [1, 1, int(block_m), int(dim)], element_bytes
            ),
            "K_desc": _descriptor(
                "K", kv_shape, [1, 1, int(block_n), int(dim)], element_bytes
            ),
            "V_desc": _descriptor(
                "V", kv_shape, [1, 1, int(block_n), int(dim)], element_bytes
            ),
        }
    if kernel_id == "tilelang-grouped-gemm":
        sizes = [int(value) for value in kwargs["batch_sizes_list"]]
        k = int(kwargs["K"])
        n = int(kwargs["N"])
        block_m, block_n, block_k = (
            int(kwargs[name]) for name in ("block_M", "block_N", "block_K")
        )
        trans_b = bool(kwargs.get("trans_b", False))
        element_bytes = _tilelang_dtype_bytes(str(kwargs.get("dtype", "float16")))
        b_shape = [len(sizes), n, k] if trans_b else [len(sizes), k, n]
        b_tile = [1, block_n, block_k] if trans_b else [1, block_k, block_n]
        return {
            "A_desc": _descriptor(
                "A", [sum(sizes), k], [block_m, block_k], element_bytes
            ),
            "B_desc": _descriptor("B", b_shape, b_tile, element_bytes),
        }
    return {}


def _tilelang_buffer_values(
    case_id: str,
    recipe: dict[str, Any],
) -> dict[str, list[int]]:
    kernel_id = case_id.split("-source-", 1)[0]
    if kernel_id != "tilelang-grouped-gemm":
        return {}
    kwargs = recipe["materialization"].get("kwargs", {})
    sizes = [int(value) for value in kwargs["batch_sizes_list"]]
    block_m = int(kwargs["block_M"])
    offsets: list[int] = []
    padded_offsets: list[int] = []
    offset = 0
    padded = 0
    for size in sizes:
        offsets.append(offset)
        padded_offsets.append(padded)
        offset += size
        padded += ((size + block_m - 1) // block_m) * block_m
    return {
        "batch_sizes": sizes,
        "batch_offsets": offsets,
        "batch_padded_offsets": padded_offsets,
    }


_TRITON_POINTER_ALLOCATIONS = {
    "triton-softmax": {"output_ptr": "output", "input_ptr": "input"},
    "triton-layer-norm": {
        "X": "x", "Y": "y", "W": "w", "B": "b", "Mean": "mean", "Rstd": "rstd",
    },
    "triton-grouped-gemm": {
        "group_a_ptrs": "a_ptrs",
        "group_b_ptrs": "b_ptrs",
        "group_c_ptrs": "c_ptrs",
        "group_gemm_sizes": "sizes",
        "g_lds": "lds",
    },
    "triton-fused-attention": {
        "M": "m", "desc_q": "q", "desc_k": "k", "desc_v": "v", "desc_o": "o",
    },
}
_TRITON_FAULT_POINTER_ALLOCATIONS = {
    "pipeguard_scratch": "pipeguard_scratch",
    "pipeguard_wide": "pipeguard_auxiliary",
}


def _triton_scalar_bindings(case_id: str, recipe: dict[str, Any]) -> dict[str, int]:
    kernel_id = case_id.split("-source-", 1)[0]
    bindings = recipe.get("bindings", {})
    if kernel_id == "triton-softmax":
        m, n = int(bindings["M"]), int(bindings["N"])
        return {"input_row_stride": n, "output_row_stride": n, "n_rows": m, "n_cols": n}
    if kernel_id == "triton-layer-norm":
        n = int(bindings["N"])
        return {"stride": n, "N": n}
    if kernel_id == "triton-grouped-gemm":
        return {"group_size": len(bindings["group_sizes"])}
    if kernel_id == "triton-fused-attention":
        return {
            "Z": int(bindings["batch"]),
            "H": int(bindings["heads"]),
            "N_CTX": int(bindings["seq_q"]),
        }
    return {}


def _triton_function_arguments(ttgir: str) -> list[tuple[str, str]] | None:
    """Ordered (loc name, type) pairs of the public TTGIR kernel arguments."""
    match = re.search(r"tt\.func public @[\w.$-]+\((.*)\)\s*attributes\s*\{", ttgir)
    if match is None:
        return None
    body = match.group(1)
    arguments = re.findall(
        r'%[\w.$-]+:\s*([^\s,{]+)(?:\s*\{[^{}]*\})?\s*loc\("([^"]+)"', body
    )
    if not arguments or len(arguments) != body.count("%"):
        return None
    return [(name, type_text) for type_text, name in arguments]


def _triton_tma_descriptor_evidence(
    raw_path: Path,
    recipe: dict[str, Any],
    argument_names: list[str],
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    from triton._C.libtriton import ir, nvidia

    bindings = recipe.get("bindings", {})
    if str(bindings.get("dtype", "")) not in {"float16", "bfloat16"}:
        return None
    element_bytes = 2
    try:
        rows = int(bindings["batch"]) * int(bindings["heads"]) * int(bindings["seq_q"])
        dim = int(bindings["dim"])
        tile_rows = {
            "desc_q": int(bindings["BLOCK_M"]),
            "desc_k": int(bindings["BLOCK_N"]),
            "desc_v": int(bindings["BLOCK_N"]),
            "desc_o": int(bindings["BLOCK_M"]),
        }
    except (KeyError, TypeError, ValueError):
        return None
    context = ir.context()
    ir.load_dialects(context)
    nvidia.load_dialects(context)
    try:
        module = ir.parse_mlir_module(str(raw_path), context)
        module.verify()
    except (OSError, RuntimeError):
        return None
    operations: list[Any] = []
    module.walk(operations.append)
    entry_block = None
    tensormap_bases: dict[int, int] = {}
    descriptor_bases: dict[int, int] = {}
    copies: list[tuple[int, int | None]] = []
    unit_attributes: dict[str, list[str]] = {}
    for index, operation in enumerate(operations):
        name = operation.get_name()
        if name == "ttng.tensormap_create" and operation.get_num_operands() >= 2:
            tensormap_bases[operation.get_operand(0).id()] = operation.get_operand(1).id()
            entry_block = operation.get_block()
        elif (
            name == "ttng.reinterpret_tensor_descriptor"
            and operation.get_num_operands() == 1
            and operation.get_num_results() == 1
        ):
            base = tensormap_bases.get(operation.get_operand(0).id())
            if base is not None:
                descriptor_bases[operation.get_result(0).id()] = base
        elif name in ("ttng.async_tma_copy_global_to_local", "ttng.async_tma_copy_local_to_global"):
            copies.append((index, descriptor_bases.get(operation.get_operand(0).id())))
        elif name == "ttng.async_tma_store_wait":
            unit_attributes[str(index)] = []
    if entry_block is None or not copies:
        return None
    argument_positions = {
        entry_block.get_argument(position).id(): position
        for position in range(entry_block.get_num_arguments())
    }
    allocations = _TRITON_POINTER_ALLOCATIONS["triton-fused-attention"]
    array_attributes: dict[str, Any] = {}
    for index, base in copies:
        position = argument_positions.get(base) if base is not None else None
        if position is None or position >= len(argument_names):
            return None
        name = argument_names[position]
        if name not in tile_rows:
            return None
        array_attributes[str(index)] = {
            "descriptor": {
                "allocation_id": f"global:{allocations[name]}",
                "element_bytes": element_bytes,
                "shape": [rows, dim],
                "strides": [dim, 1],
                "tile_shape": [tile_rows[name], dim],
                "coordinate_order": [1, 0],
            },
        }
    return array_attributes, unit_attributes


def _triton_binding_evidence(
    raw_path: Path,
    recipe: dict[str, Any],
    case_id: str,
) -> dict[str, Any] | None:
    kernel_id = case_id.split("-source-", 1)[0]
    pointer_names = _TRITON_POINTER_ALLOCATIONS.get(kernel_id)
    scalar_bindings = _triton_scalar_bindings(case_id, recipe)
    if pointer_names is None or not scalar_bindings:
        return None
    arguments = _triton_function_arguments(raw_path.read_text(encoding="utf-8"))
    if arguments is None:
        return None
    signature = recipe["compiler"].get("signature", {})
    expected = {name for name, kind in signature.items() if kind != "constexpr"}
    if {name for name, _ in arguments} != expected:
        return None
    pointer_names = {**pointer_names, **_TRITON_FAULT_POINTER_ALLOCATIONS}
    scalar_values: dict[str, int] = {}
    pointer_allocations: dict[str, str] = {}
    for index, (name, type_text) in enumerate(arguments):
        if type_text.startswith("!tt.ptr<"):
            if name in pointer_names:
                pointer_allocations[str(index)] = f"global:{pointer_names[name]}"
        elif name in scalar_bindings and re.fullmatch(r"(?:i|ui)\d+|index", type_text):
            scalar_values[str(index)] = scalar_bindings[name]
    if len(scalar_values) != len(scalar_bindings):
        return None
    recipe_allocations = recipe["compiler"].get("binding_evidence", {}).get("pointer_allocations", {})
    if recipe_allocations and set(pointer_allocations.values()) != set(recipe_allocations.values()):
        return None
    array_attributes: dict[str, Any] = {}
    unit_attributes: dict[str, Any] = {}
    if kernel_id == "triton-fused-attention":
        descriptors = _triton_tma_descriptor_evidence(
            raw_path, recipe, [name for name, _ in arguments]
        )
        if descriptors is not None:
            array_attributes, unit_attributes = descriptors
    return {
        "scalar_values": scalar_values,
        "pointer_allocations": pointer_allocations,
        "array_attributes": array_attributes,
        "unit_attributes": unit_attributes,
    }


def _diagnostic(code: str, message: str, **details: Any) -> dict[str, Any]:
    result: dict[str, Any] = {"code": code, "message": message}
    if details:
        result["details"] = details
    return result


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _write_json(path: Path, value: Any) -> dict[str, str]:
    atomic_write(path, pretty_json(value))
    return {"path": path.name, "sha256": sha256_file(path)}


def _write_text(path: Path, value: str) -> dict[str, str]:
    atomic_write(path, value)
    return {"path": path.name, "sha256": sha256_file(path)}


def _specialize_tilelang_module(tvm: Any, module: Any, recipe: dict[str, Any]) -> Any:
    scalar_values = recipe["compiler"].get("scalar_specializations", {})
    functions = {}
    for global_var, function in module.functions.items():
        parameter_map = {
            parameter: scalar_values[parameter.name]
            for parameter in function.params
            if parameter.name in scalar_values
        }
        if parameter_map:
            function = function.specialize(parameter_map)
        functions[global_var] = function
    return tvm.IRModule(functions)


def _capture_tilelang(raw_path: Path, recipe: dict[str, Any], case_id: str) -> dict[str, Any]:
    import tilelang_capture

    compiler = recipe["compiler"]
    source_digest = sha256_file(raw_path)
    descriptors = _tilelang_descriptor_evidence(case_id, recipe)
    descriptor_path = None
    if descriptors:
        descriptor_path = raw_path.with_name(f"{raw_path.stem}.descriptor-evidence.json")
        atomic_write(descriptor_path, pretty_json({
            "evidence_version": "tilelang-tma-descriptor-evidence-v1",
            "source_ir_sha256": source_digest,
            "descriptors": descriptors,
        }))
    buffer_values = _tilelang_buffer_values(case_id, recipe)
    binding_path = None
    if buffer_values:
        binding_path = raw_path.with_name(f"{raw_path.stem}.binding-evidence.json")
        atomic_write(binding_path, pretty_json({
            "evidence_version": "tilelang-binding-evidence-v1",
            "source_ir_sha256": source_digest,
            "scalar_values": {},
            "pointer_table_allocations": {},
            "buffer_values": buffer_values,
            "provenance": {
                "kind": "SUITE4_RECIPE",
                "case_id": case_id,
            },
        }))
    return tilelang_capture.capture_file(
        raw_path,
        source=f"suite4:{case_id}",
        command="suite4-native-tilelang-compile-only",
        descriptor_evidence_path=descriptor_path,
        binding_evidence_path=binding_path,
        launch_grid=compiler["grid"],
        selected_cta=tuple(compiler["selected_cta"]),
    )


def _capture_triton(raw_path: Path, recipe: dict[str, Any], case_id: str, output: Path) -> dict[str, Any]:
    import triton_capture

    compiler = recipe["compiler"]
    evidence = compiler.get("binding_evidence")
    evidence_path = None
    if evidence is not None:
        evidence = dict(evidence)
        if not evidence.get("scalar_values"):
            derived = _triton_binding_evidence(raw_path, recipe, case_id)
            if derived is not None:
                evidence["scalar_values"] = derived["scalar_values"]
                evidence["pointer_allocations"] = derived["pointer_allocations"]
                evidence["array_attributes"] = {
                    **derived["array_attributes"], **evidence.get("array_attributes", {}),
                }
                evidence["unit_attributes"] = {
                    **derived["unit_attributes"], **evidence.get("unit_attributes", {}),
                }
        evidence["source_ir_sha256"] = sha256_file(raw_path)
        evidence_path = output / "binding-evidence.json"
        atomic_write(evidence_path, pretty_json(evidence))
    return triton_capture.capture_file(
        raw_path,
        source=f"suite4:{case_id}",
        command="suite4-native-triton-compile-only",
        grid_ctas=compiler.get("grid_ctas", compiler["grid"][0]),
        binding_evidence_path=evidence_path,
        launch_grid=compiler["grid"],
        selected_cta=tuple(compiler["selected_cta"]),
    )


def _parse_and_check(dsl: str, snapshot_path: Path, pgir_path: Path, metadata_path: Path, report_path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    environment = {
        **os.environ,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONHASHSEED": "0",
        "PYTHONPATH": os.pathsep.join((str(ROOT / "src"), str(ROOT / "tools"))),
    }
    subprocess.run(
        [
            "/usr/bin/python3", str(ROOT / "tools/upstream_parse.py"),
            "--dsl", dsl, "--snapshot", str(snapshot_path),
            "--pgir", str(pgir_path), "--metadata", str(metadata_path),
        ],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    checker_request_path = report_path.with_suffix(".request.json")
    atomic_write(checker_request_path, pretty_json({
        "pgir_path": str(pgir_path), "engine": "maximal", "report_path": str(report_path),
    }))
    subprocess.run(
        ["/usr/bin/python3", str(ROOT / "tools/upstream_check.py"), "--worker-request", str(checker_request_path)],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    return load_json(metadata_path), load_json(report_path)


def run(request: dict[str, Any], output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    result: dict[str, Any] = {
        "schema_version": "pipeguard-mutation-run-result-v1",
        "fingerprint": request["fingerprint"],
        "case_id": request["case"]["case_id"],
        "kernel_id": request["case"]["kernel_id"],
        "variant_id": request["case"]["variant_id"],
        "spec_id": request["case"]["spec_id"],
        "dsl": request["dsl"],
        "pipeline_status": "INFRA_ERROR",
        "checker_status": "NOT_RUN",
        "numeric_status": "NOT_RUN",
        "diagnostics": [],
        "artifacts": {},
        "timings": {},
    }
    source_path = Path(request["source_path"])
    recipe = request["recipe"]
    active_recipe = recipe
    try:
        phase = time.monotonic()
        module = _load_module(source_path, f"pipeguard_suite4_{request['fingerprint'][:16]}")
        result["timings"]["import"] = time.monotonic() - phase
    except Exception as error:
        result["pipeline_status"] = "IMPORT_ERROR"
        result["diagnostics"].append(_diagnostic(
            "SUITE4_IMPORT_FAILED", str(error), error_type=type(error).__name__, traceback=traceback.format_exc(limit=20)
        ))
        result["timings"]["total"] = time.monotonic() - started
        return result

    try:
        phase = time.monotonic()
        raw_paths: list[tuple[str, Path]] = []
        if request["dsl"] == "tilelang":
            from tilelang import tvm

            lowering_recipe = json.loads(json.dumps(recipe))
            materialization_kwargs = lowering_recipe["materialization"]["kwargs"]
            for name in lowering_recipe["compiler"].get("tuple_materialization_kwargs", []):
                materialization_kwargs[name] = tuple(materialization_kwargs[name])
            _tvm, modules = _lower_tilelang(module, lowering_recipe, resolution="capsule")
            modules = [
                (symbol, _specialize_tilelang_module(tvm, device_module, recipe))
                for symbol, device_module in modules
            ]
            for index, (symbol, device_module) in enumerate(modules):
                path = output / ("raw.tvm.json" if len(modules) == 1 else f"raw-{index:03d}-{symbol}.tvm.json")
                result["artifacts"][f"raw:{symbol}"] = _write_text(path, tvm.ir.save_json(device_module))
                result.setdefault("lowered_semantic_sha256", {})[symbol] = hashlib.sha256(
                    str(tvm.ir.structural_hash(device_module, map_free_vars=False)).encode("utf-8")
                ).hexdigest()
                raw_paths.append((symbol, path))
        else:
            active_recipe = augment_triton_recipe(recipe, request["case"]["variant_id"])
            text = _lower_triton(module, active_recipe, resolution="capsule")
            path = output / "raw.ttgir.mlir"
            result["artifacts"]["raw:kernel"] = _write_text(path, text)
            result.setdefault("lowered_semantic_sha256", {})["kernel"] = hashlib.sha256(
                _normalize_triton_ir(text).encode("utf-8")
            ).hexdigest()
            raw_paths.append(("kernel", path))
        result["timings"]["lowering"] = time.monotonic() - phase
    except Exception as error:
        result["pipeline_status"] = "LOWERING_ERROR"
        result["diagnostics"].append(_diagnostic(
            "SUITE4_LOWERING_FAILED", str(error), error_type=type(error).__name__, traceback=traceback.format_exc(limit=30)
        ))
        result["timings"]["total"] = time.monotonic() - started
        return result

    captures = 0
    for index, (symbol, raw_path) in enumerate(raw_paths):
        try:
            phase = time.monotonic()
            snapshot = (
                _capture_tilelang(raw_path, active_recipe, result["case_id"])
                if request["dsl"] == "tilelang"
                else _capture_triton(raw_path, active_recipe, result["case_id"], output)
            )
            snapshot_path = output / ("snapshot.json" if len(raw_paths) == 1 else f"snapshot-{index:03d}-{symbol}.json")
            result["artifacts"][f"snapshot:{symbol}"] = _write_json(snapshot_path, snapshot)
            pgir_path = output / ("pgir.json" if len(raw_paths) == 1 else f"pgir-{index:03d}-{symbol}.json")
            metadata_path = output / ("pgir-metadata.json" if len(raw_paths) == 1 else f"pgir-metadata-{index:03d}-{symbol}.json")
            report_path = output / ("checker.json" if len(raw_paths) == 1 else f"checker-{index:03d}-{symbol}.json")
            metadata, report = _parse_and_check(
                request["dsl"], snapshot_path, pgir_path, metadata_path, report_path
            )
            pgir_ref = {"path": pgir_path.name, "sha256": sha256_file(pgir_path), "semantic_sha256": metadata["semantic_sha256"]}
            result["artifacts"][f"pgir:{symbol}"] = pgir_ref
            result["artifacts"][f"checker:{symbol}"] = {"path": report_path.name, "sha256": sha256_file(report_path)}
            result.setdefault("checker_verdicts", []).append(report["verdict"])
            captures += 1
            result["timings"][f"capture_check:{symbol}"] = time.monotonic() - phase
        except Exception as error:
            diagnostic = error.diagnostic() if hasattr(error, "diagnostic") else _diagnostic(
                "SUITE4_CAPTURE_OR_CHECK_FAILED", str(error), error_type=type(error).__name__, traceback=traceback.format_exc(limit=30)
            )
            result["diagnostics"].append(diagnostic)
    if captures == len(raw_paths):
        result["pipeline_status"] = "CAPTURED"
        result["checker_status"] = "CHECKED"
    elif captures:
        result["pipeline_status"] = "UNSUPPORTED"
        result["checker_status"] = "CHECKED"
        result["diagnostics"].append(_diagnostic("SUITE4_PARTIAL_CAPTURE", "Only part of the lowered device module was captured", captured=captures, total=len(raw_paths)))
    else:
        result["pipeline_status"] = "UNSUPPORTED"
        result["checker_status"] = "CAPTURE_NOT_AVAILABLE"
    result["timings"]["total"] = time.monotonic() - started
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = load_json(args.request)
    result = run(request, args.output)
    atomic_write(args.output / "result.json", pretty_json(result))
    print(json.dumps({"case_id": result["case_id"], "pipeline_status": result["pipeline_status"], "checker_status": result["checker_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
