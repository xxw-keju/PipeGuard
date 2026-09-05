from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import Any, Protocol

from upstream_static_eval import json_literal_value, literal_value


class AliasResolver(Protocol):
    def qualified(self, node: ast.AST) -> str | None: ...


@dataclass(frozen=True)
class TritonArgumentBinding:
    signature: dict[str, str]
    constexprs: dict[str, Any]
    attrs: dict[str, list[list[Any]]]


_TYPE_NAMES = {
    "bool": "u1",
    "int1": "u1",
    "uint1": "u1",
    "i1": "u1",
    "float8e4nv": "fp8e4nv",
    "float8e5": "fp8e5",
    "float8e4b15": "fp8e4b15",
    "float8_e4m3fn": "fp8e4nv",
    "float8e4b8": "fp8e4b8",
    "float8_e4m3fnuz": "fp8e4b8",
    "float8_e5m2": "fp8e5",
    "float8e5b16": "fp8e5b16",
    "float8_e5m2fnuz": "fp8e5b16",
    "half": "fp16",
    "float16": "fp16",
    "bfloat16": "bf16",
    "float": "fp32",
    "float32": "fp32",
    "double": "fp64",
    "float64": "fp64",
    "int8": "i8",
    "int16": "i16",
    "int": "i32",
    "int32": "i32",
    "int64": "i64",
    "uint8": "u8",
    "uint16": "u16",
    "uint32": "u32",
    "uint64": "u64",
    "void": "void",
}
_TYPE_NAMES.update({value: value for value in tuple(_TYPE_NAMES.values())})
_TORCH_DTYPES = {
    "torch.bool": "u1",
    "torch.float8_e4m3fn": "fp8e4nv",
    "torch.float8_e4m3fnuz": "fp8e4b8",
    "torch.float8_e5m2": "fp8e5",
    "torch.float8_e5m2fnuz": "fp8e5b16",
    "torch.float16": "fp16",
    "torch.half": "fp16",
    "torch.bfloat16": "bf16",
    "torch.float32": "fp32",
    "torch.float": "fp32",
    "torch.float64": "fp64",
    "torch.double": "fp64",
    "torch.int8": "i8",
    "torch.int16": "i16",
    "torch.int32": "i32",
    "torch.int": "i32",
    "torch.int64": "i64",
    "torch.long": "i64",
    "torch.uint8": "u8",
}


def _is_constexpr(annotation: ast.AST | None, aliases: AliasResolver) -> bool:
    if annotation is None:
        return False
    if aliases.qualified(annotation) == "triton.language.constexpr":
        return True
    return isinstance(annotation, ast.Constant) and isinstance(annotation.value, str) and annotation.value in {
        "constexpr",
        "tl.constexpr",
        "triton.language.constexpr",
    }


def _normalized_annotation(annotation: ast.AST | None, aliases: AliasResolver) -> str | None:
    if annotation is None or _is_constexpr(annotation, aliases):
        return None
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        text = annotation.value.strip()
    else:
        qualified = aliases.qualified(annotation)
        if qualified is None:
            return None
        text = qualified.removeprefix("triton.language.")
    const = False
    if text.startswith("const "):
        const = True
        text = text.removeprefix("const ").strip()
    pointer = text.startswith("*") or text.endswith("*")
    if text.startswith("*"):
        text = text[1:]
    if text.endswith("*"):
        text = text[:-1]
    text = text.removeprefix("tl.").replace("_t", "")
    canonical = _TYPE_NAMES.get(text)
    if canonical is None:
        return None
    if pointer:
        return ("*k" if const else "*") + canonical
    return canonical


def _literal_option(
    candidate: Any,
    aliases: AliasResolver,
    name: str,
) -> set[str | int] | None:
    decorators = [
        decorator
        for decorator in candidate.node.decorator_list
        if aliases.qualified(decorator.func if isinstance(decorator, ast.Call) else decorator) == "triton.jit"
    ]
    if len(decorators) != 1:
        return None
    decorator = decorators[0]
    if not isinstance(decorator, ast.Call):
        return set()
    if decorator.args:
        return None
    values = [keyword.value for keyword in decorator.keywords if keyword.arg == name]
    if len(values) > 1 or any(keyword.arg is None for keyword in decorator.keywords):
        return None
    if not values:
        return set()
    try:
        value = literal_value(values[0], {})
    except ValueError:
        return None
    if isinstance(value, (str, int)) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, (str, int)) or isinstance(item, bool) for item in value
    ):
        return None
    return set(value)


def _specialization_flags(
    candidate: Any,
    aliases: AliasResolver,
    parameters: tuple[ast.arg, ...],
) -> tuple[set[int], set[int]] | None:
    disabled = _literal_option(candidate, aliases, "do_not_specialize")
    alignment_disabled = _literal_option(candidate, aliases, "do_not_specialize_on_alignment")
    if disabled is None or alignment_disabled is None:
        return None
    names = {parameter.arg: index for index, parameter in enumerate(parameters)}

    def indices(values: set[str | int]) -> set[int] | None:
        result: set[int] = set()
        for value in values:
            index = names.get(value) if isinstance(value, str) else value
            if not isinstance(index, int) or not 0 <= index < len(parameters):
                return None
            result.add(index)
        return result

    disabled_indices = indices(disabled)
    alignment_indices = indices(alignment_disabled)
    if disabled_indices is None or alignment_indices is None:
        return None
    return disabled_indices, alignment_indices


def _integer_type(value: int) -> str | None:
    if -(2**31) <= value <= 2**31 - 1:
        return "i32"
    if -(2**63) <= value <= 2**63 - 1:
        return "i64"
    if 2**63 <= value <= 2**64 - 1:
        return "u64"
    return None


def _attrs(key: str | None) -> list[list[Any]] | None:
    if key is None:
        return None
    if key == "":
        return []
    if key == "D":
        return [["tt.divisibility", 16]]
    raise AssertionError(f"unexpected CUDA specialization key: {key}")


def _specialize_scalar(
    value: Any,
    annotation: str | None,
    *,
    specialize_value: bool,
    align: bool,
) -> tuple[str, Any | None, list[list[Any]] | None] | None:
    if value is None:
        return "constexpr", None, None
    if isinstance(value, bool):
        inferred = "u1"
        key = None
    elif isinstance(value, int):
        inferred = _integer_type(value)
        if inferred is None:
            return None
        if value == 1 and specialize_value:
            return "constexpr", 1, None
        key = None if not specialize_value else ("D" if align and value % 16 == 0 else "")
    elif isinstance(value, float):
        inferred = "fp32"
        key = None
    else:
        return None
    result_type = annotation or inferred
    if annotation == "u1" or (annotation is not None and annotation.startswith(("fp", "bf"))):
        key = None
    return result_type, None, _attrs(key)


def _warmup_dtype(
    node: ast.AST,
    aliases: AliasResolver,
    annotation: str | None,
    *,
    specialize_value: bool,
    align: bool,
) -> tuple[str, Any | None, list[list[Any]] | None] | None:
    dtype = _TORCH_DTYPES.get(aliases.qualified(node) or "")
    if dtype is None:
        return None
    inferred = f"*{dtype}"
    result_type = annotation or inferred
    if not result_type.startswith(("*",)):
        return None
    key = None if not specialize_value else ("D" if align else "")
    return result_type, None, _attrs(key)


def bind_triton_arguments(
    candidate: Any,
    call: ast.Call,
    environment: dict[str, Any],
    aliases: AliasResolver,
    keywords: dict[str, ast.AST],
    ignored_keywords: set[str],
    *,
    warmup: bool,
) -> tuple[TritonArgumentBinding | None, str]:
    arguments = candidate.node.args
    if arguments.vararg is not None or arguments.kwarg is not None:
        return None, "VARARGS_UNSUPPORTED"
    positional = (*arguments.posonlyargs, *arguments.args)
    parameters = (*positional, *arguments.kwonlyargs)
    flags = _specialization_flags(candidate, aliases, parameters)
    if flags is None:
        return None, "SPECIALIZATION_OPTIONS_UNKNOWN"
    disabled, alignment_disabled = flags
    if any(isinstance(value, ast.Starred) for value in call.args) or len(call.args) > len(positional):
        return None, "BINDING_UNKNOWN"
    bound: dict[str, tuple[ast.AST, bool]] = {
        parameter.arg: (value, warmup)
        for parameter, value in zip(positional, call.args)
    }
    parameter_names = {parameter.arg for parameter in parameters}
    positional_only = {parameter.arg for parameter in arguments.posonlyargs}
    for name, value in keywords.items():
        if name in ignored_keywords:
            continue
        if name not in parameter_names or name in bound or name in positional_only:
            return None, "BINDING_UNKNOWN"
        bound[name] = (value, False)
    defaults: dict[str, ast.AST] = {}
    if arguments.defaults:
        defaults.update({
            parameter.arg: value
            for parameter, value in zip(positional[-len(arguments.defaults):], arguments.defaults)
        })
    defaults.update({
        parameter.arg: value
        for parameter, value in zip(arguments.kwonlyargs, arguments.kw_defaults)
        if value is not None
    })
    signature: dict[str, str] = {}
    constexprs: dict[str, Any] = {}
    attrs: dict[str, list[list[Any]]] = {}
    for index, parameter in enumerate(parameters):
        pair = bound.get(parameter.arg)
        if pair is None:
            default = defaults.get(parameter.arg)
            if default is None:
                return None, "BINDING_UNKNOWN"
            pair = (default, False)
        node, wrap_dtype = pair
        if _is_constexpr(parameter.annotation, aliases):
            try:
                value = json_literal_value(node, environment)
            except ValueError:
                return None, "CONSTEXPR_BINDING_UNKNOWN"
            signature[parameter.arg] = "constexpr"
            constexprs[parameter.arg] = value
            continue
        annotation = _normalized_annotation(parameter.annotation, aliases)
        if parameter.annotation is not None and annotation is None:
            return None, "ANNOTATION_UNSUPPORTED"
        specialized = None
        if wrap_dtype:
            specialized = _warmup_dtype(
                node,
                aliases,
                annotation,
                specialize_value=index not in disabled,
                align=index not in alignment_disabled,
            )
        if specialized is None:
            try:
                value = literal_value(node, environment)
            except ValueError:
                return None, "RUNTIME_VALUE_UNKNOWN"
            specialized = _specialize_scalar(
                value,
                annotation,
                specialize_value=index not in disabled,
                align=index not in alignment_disabled,
            )
        if specialized is None:
            return None, "RUNTIME_TYPE_UNKNOWN"
        type_name, constant, argument_attrs = specialized
        signature[parameter.arg] = type_name
        if type_name == "constexpr":
            constexprs[parameter.arg] = constant
        elif argument_attrs is not None:
            attrs[parameter.arg] = argument_attrs
    return TritonArgumentBinding(signature, constexprs, attrs), "ACCEPTED"
