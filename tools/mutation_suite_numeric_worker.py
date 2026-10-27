from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from mutation_suite_triton_faults import allocate_fault_buffers  # noqa: E402
from upstream_common import atomic_write, pretty_json  # noqa: E402


def _load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load source module {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _unwrap_triton(value: Any) -> Any:
    while not hasattr(value, "src") and hasattr(value, "fn"):
        value = value.fn
    return value


def _dtype(torch: Any, name: str) -> Any:
    return getattr(torch, name.removeprefix("torch."))


def _compare(torch: Any, actual: Any, expected: Any, atol: float, rtol: float) -> dict[str, Any]:
    actual_values = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected_values = expected if isinstance(expected, (tuple, list)) else (expected,)
    if len(actual_values) != len(expected_values):
        raise AssertionError(f"output count differs: {len(actual_values)} != {len(expected_values)}")
    metrics = []
    for got, want in zip(actual_values, expected_values):
        got_float = got.detach().float()
        want_float = want.detach().float()
        delta = (got_float - want_float).abs()
        metrics.append({
            "shape": list(got.shape),
            "max_abs_error": float(delta.max().item()) if delta.numel() else 0.0,
            "mean_abs_error": float(delta.mean().item()) if delta.numel() else 0.0,
        })
        torch.testing.assert_close(got, want, atol=atol, rtol=rtol)
    return {"outputs": metrics}


def _compile_tilelang(module: Any, recipe: dict[str, Any]) -> Any:
    materialization = recipe["materialization"]
    factory = module.pipeguard_resolve(**materialization["resolver_kwargs"])
    kwargs = dict(materialization["kwargs"])
    for name in recipe["compiler"].get("tuple_materialization_kwargs", []):
        kwargs[name] = tuple(kwargs[name])
    return factory.compile(*materialization["args"], **kwargs)


def _dense_gemm(module: Any, recipe: dict[str, Any], torch: Any, variant_id: str) -> tuple[Any, Any]:
    b = recipe["bindings"]
    dtype = _dtype(torch, b["dtype"])
    a = torch.randn((b["M"], b["K"]), device="cuda", dtype=dtype)
    rhs = torch.randn((b["K"], b["N"]), device="cuda", dtype=dtype)
    reference = torch.matmul(a, rhs)
    if recipe["dsl"] == "tilelang":
        actual = _compile_tilelang(module, recipe)(a, rhs)
    else:
        out = torch.empty((b["M"], b["N"]), device="cuda", dtype=dtype)
        fault = allocate_fault_buffers(torch, variant_id, recipe["compiler"]["grid"][0])
        arguments = [a, rhs, out]
        if fault is not None:
            arguments.extend(fault)
        arguments.extend([
            b["M"], b["N"], b["K"], a.stride(0), a.stride(1), rhs.stride(0),
            rhs.stride(1), out.stride(0), out.stride(1),
        ])
        kernel = _unwrap_triton(module.pipeguard_resolve())
        kernel[tuple(recipe["compiler"]["grid"])](
            *arguments,
            BLOCK_SIZE_M=b["BLOCK_M"], BLOCK_SIZE_N=b["BLOCK_N"], BLOCK_SIZE_K=b["BLOCK_K"],
            GROUP_SIZE_M=8, ACTIVATION="", num_warps=4, num_stages=3,
        )
        actual = out
    return actual, reference


def _grouped_gemm(module: Any, recipe: dict[str, Any], torch: Any, variant_id: str) -> tuple[Any, Any]:
    b = recipe["bindings"]
    dtype = _dtype(torch, b["dtype"])
    sizes = list(b["group_sizes"])
    a_list = [torch.randn((m, b["K"]), device="cuda", dtype=dtype) for m in sizes]
    b_list = [torch.randn((b["K"], b["N"]), device="cuda", dtype=dtype) for _ in sizes]
    references = [torch.matmul(a, rhs) for a, rhs in zip(a_list, b_list)]
    if recipe["dsl"] == "tilelang":
        a = torch.cat(a_list, dim=0)
        rhs = torch.stack(b_list)
        sizes_tensor = torch.tensor(sizes, device="cuda", dtype=torch.int32)
        offsets = torch.tensor([sum(sizes[:index]) for index in range(len(sizes))], device="cuda", dtype=torch.int32)
        padded_sizes = [((size + b["BLOCK_M"] - 1) // b["BLOCK_M"]) * b["BLOCK_M"] for size in sizes]
        padded_offsets = torch.tensor([sum(padded_sizes[:index]) for index in range(len(sizes))], device="cuda", dtype=torch.int32)
        actual = _compile_tilelang(module, recipe)(a, rhs, sizes_tensor, offsets, padded_offsets)
        return actual, torch.cat(references, dim=0)

    outputs = [torch.empty((m, b["N"]), device="cuda", dtype=dtype) for m in sizes]
    a_ptrs = torch.tensor([value.data_ptr() for value in a_list], device="cuda", dtype=torch.uint64)
    b_ptrs = torch.tensor([value.data_ptr() for value in b_list], device="cuda", dtype=torch.uint64)
    c_ptrs = torch.tensor([value.data_ptr() for value in outputs], device="cuda", dtype=torch.uint64)
    gemm_sizes = torch.tensor([[m, b["N"], b["K"]] for m in sizes], device="cuda", dtype=torch.int32)
    lds = torch.tensor([[b["K"], b["N"], b["N"]] for _ in sizes], device="cuda", dtype=torch.int32)
    slots = sum(
        ((m + b["BLOCK_M"] - 1) // b["BLOCK_M"]) * ((b["N"] + b["BLOCK_N"] - 1) // b["BLOCK_N"])
        for m in sizes
    )
    fault = allocate_fault_buffers(torch, variant_id, slots)
    arguments = [a_ptrs, b_ptrs, c_ptrs]
    if fault is not None:
        arguments.extend(fault)
    arguments.extend([gemm_sizes, lds, len(sizes)])
    kernel = _unwrap_triton(module.pipeguard_resolve())
    kernel[tuple(recipe["compiler"]["grid"])](
        *arguments,
        NUM_SM=recipe["compiler"]["constexprs"]["NUM_SM"],
        BLOCK_SIZE_M=b["BLOCK_M"], BLOCK_SIZE_N=b["BLOCK_N"], BLOCK_SIZE_K=b["BLOCK_K"],
        num_warps=4, num_stages=3,
    )
    return tuple(outputs), tuple(references)


def _attention_reference(torch: Any, q: Any, k: Any, v: Any, causal: bool) -> Any:
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    if causal:
        q_len, kv_len = q.shape[-2], k.shape[-2]
        past = kv_len - q_len
        q_index = torch.arange(q_len, device=q.device)[:, None] + past
        k_index = torch.arange(kv_len, device=q.device)[None, :]
        scores = scores.masked_fill(q_index < k_index, float("-inf"))
    return torch.matmul(torch.softmax(scores, dim=-1), v.float()).to(q.dtype)


def _fused_attention(module: Any, recipe: dict[str, Any], torch: Any, variant_id: str) -> tuple[Any, Any]:
    b = recipe["bindings"]
    dtype = _dtype(torch, b["dtype"])
    shape_q = (b["batch"], b["heads"], b["seq_q"], b["dim"])
    shape_kv = (b["batch"], b["heads"], b["seq_kv"], b["dim"])
    q = torch.randn(shape_q, device="cuda", dtype=dtype)
    k = torch.randn(shape_kv, device="cuda", dtype=dtype)
    v = torch.randn(shape_kv, device="cuda", dtype=dtype)
    reference = _attention_reference(torch, q, k, v, b["is_causal"])
    if recipe["dsl"] == "tilelang":
        actual = _compile_tilelang(module, recipe)(q, k, v)
        if isinstance(actual, (tuple, list)):
            actual = actual[0]
        return actual, reference

    import triton

    triton.set_allocator(lambda size, alignment, stream: torch.empty(size, dtype=torch.int8, device="cuda"))
    out = torch.empty_like(q)
    m_buffer = torch.empty((b["batch"], b["heads"], b["seq_q"]), device="cuda", dtype=torch.float32)
    fault = allocate_fault_buffers(
        torch, variant_id, recipe["compiler"]["grid"][0] * recipe["compiler"]["grid"][1]
    )
    arguments = [b["dim"] ** -0.5, m_buffer, b["batch"], b["heads"], q, k, v, out]
    if fault is not None:
        arguments.extend(fault)
    arguments.append(b["seq_kv"])
    kernel = _unwrap_triton(module.pipeguard_resolve())
    kernel[tuple(recipe["compiler"]["grid"])](
        *arguments,
        HEAD_DIM=b["dim"], BLOCK_M=b["BLOCK_M"], BLOCK_N=b["BLOCK_N"], FP8_OUTPUT=False,
        STAGE=3 if b["is_causal"] else 1, warp_specialize=False, IS_HOPPER=True,
        num_warps=4, num_stages=3,
    )
    return out, reference


def _layer_norm(module: Any, recipe: dict[str, Any], torch: Any, variant_id: str) -> tuple[Any, Any]:
    b = recipe["bindings"]
    dtype = _dtype(torch, b["dtype"])
    x = torch.randn((b["M"], b["N"]), device="cuda", dtype=dtype)
    weight = torch.randn((b["N"],), device="cuda", dtype=dtype)
    bias = torch.randn((b["N"],), device="cuda", dtype=dtype)
    reference_y = torch.nn.functional.layer_norm(x, (b["N"],), weight, bias, b["eps"])
    xf = x.float()
    reference_mean = xf.mean(dim=1)
    reference_rstd = torch.rsqrt(xf.var(dim=1, unbiased=False) + b["eps"])
    if recipe["dsl"] == "tilelang":
        actual = _compile_tilelang(module, recipe)(x, weight, bias)
        return actual, (reference_y, reference_mean, reference_rstd)

    y = torch.empty_like(x)
    mean = torch.empty((b["M"],), device="cuda", dtype=torch.float32)
    rstd = torch.empty_like(mean)
    block = recipe["compiler"]["constexprs"]["BLOCK_SIZE"]
    fault = allocate_fault_buffers(torch, variant_id, b["M"])
    arguments = [x, y]
    if fault is not None:
        arguments.extend(fault)
    arguments.extend([weight, bias, mean, rstd, x.stride(0), b["N"], b["eps"]])
    kernel = _unwrap_triton(module.pipeguard_resolve())
    kernel[tuple(recipe["compiler"]["grid"])](
        *arguments, BLOCK_SIZE=block, num_warps=4, num_stages=3,
    )
    return (y, mean, rstd), (reference_y, reference_mean, reference_rstd)


def _softmax(module: Any, recipe: dict[str, Any], torch: Any, variant_id: str) -> tuple[Any, Any]:
    b = recipe["bindings"]
    dtype = _dtype(torch, b["dtype"])
    x = torch.randn((b["M"], b["N"]), device="cuda", dtype=dtype)
    reference = torch.softmax(x, dim=1)
    if recipe["dsl"] == "tilelang":
        return _compile_tilelang(module, recipe)(x), reference

    output = torch.empty_like(x)
    fault = allocate_fault_buffers(torch, variant_id, b["M"])
    arguments = [output, x]
    if fault is not None:
        arguments.extend(fault)
    arguments.extend([x.stride(0), output.stride(0), b["M"], b["N"]])
    kernel = _unwrap_triton(module.pipeguard_resolve())
    kernel[tuple(recipe["compiler"]["grid"])](
        *arguments,
        BLOCK_SIZE=recipe["compiler"]["constexprs"]["BLOCK_SIZE"],
        num_stages=recipe["compiler"]["constexprs"]["num_stages"], num_warps=4,
    )
    return output, reference


RUNNERS = {
    "triton-dense-gemm": _dense_gemm,
    "tilelang-dense-gemm": _dense_gemm,
    "triton-grouped-gemm": _grouped_gemm,
    "tilelang-grouped-gemm": _grouped_gemm,
    "triton-fused-attention": _fused_attention,
    "tilelang-fused-attention": _fused_attention,
    "triton-layer-norm": _layer_norm,
    "tilelang-layer-norm": _layer_norm,
    "triton-softmax": _softmax,
    "tilelang-softmax": _softmax,
}


def run(request: dict[str, Any], output: Path) -> dict[str, Any]:
    import torch

    result = {
        "schema_version": "pipeguard-mutation-numeric-result-v1", "fingerprint": request["fingerprint"],
        "case_id": request["case_id"], "numeric_status": "INFRA_ERROR", "diagnostics": [], "metrics": {},
    }
    phase_path = output / "numeric-phase.json"
    try:
        torch.manual_seed(request["recipe"]["numeric"]["seed"])
        module = _load_module(Path(request["source_path"]), f"pipeguard_suite4_numeric_{request['fingerprint'][:16]}")
        atomic_write(phase_path, pretty_json({"phase": "launch"}))
        started = time.monotonic()
        actual, expected = RUNNERS[request["kernel_id"]](
            module, request["recipe"], torch, request.get("variant_id", "")
        )
        torch.cuda.synchronize()
        result["elapsed_seconds"] = time.monotonic() - started
        atomic_write(phase_path, pretty_json({"phase": "compare"}))
        result["metrics"] = _compare(
            torch, actual, expected, request["recipe"]["numeric"]["atol"], request["recipe"]["numeric"]["rtol"],
        )
        result["numeric_status"] = "PASS"
    except AssertionError as error:
        result["numeric_status"] = "NUMERIC_MISMATCH"
        result["diagnostics"].append({"code": "SUITE4_NUMERIC_MISMATCH", "message": str(error)})
    except Exception as error:
        result["numeric_status"] = "COMPILE_ERROR"
        result["diagnostics"].append({
            "code": "SUITE4_NUMERIC_EXECUTION_FAILED", "message": str(error),
            "details": {"error_type": type(error).__name__, "traceback": traceback.format_exc(limit=30)},
        })
    atomic_write(output / "numeric.json", pretty_json(result))
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    result = run(request, args.output)
    print(json.dumps({"case_id": result["case_id"], "numeric_status": result["numeric_status"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
