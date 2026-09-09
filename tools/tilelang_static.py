from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

ScalarEvaluator = Callable[[Any], int | None]
CallName = Callable[[Any], str | None]


def dtype_bit_width(dtype: Any) -> int | None:
    """Return bits per logical value, including fixed vector lanes."""

    try:
        bits = int(dtype.bits)
        lanes = int(dtype.lanes)
    except (AttributeError, TypeError, ValueError):
        text = str(dtype)
        element = re.search(r"(?:bool|(?:u?int|float|bfloat)(\d+))", text)
        if element is None:
            return None
        bits = 1 if element.group(0) == "bool" else int(element.group(1))
        lane = re.search(r"x(\d+)(?:\"|$)", text)
        lanes = int(lane.group(1)) if lane is not None else 1
    if bits <= 0 or lanes <= 0:
        return None
    return bits * lanes


def dtype_byte_width(dtype: Any) -> int | None:
    bits = dtype_bit_width(dtype)
    return bits // 8 if bits is not None and bits % 8 == 0 else None


def cp_async_transfer_bytes(count: int, destination_dtype: Any, source_dtype: Any) -> int | None:
    if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
        return None
    destination_bits = dtype_bit_width(destination_dtype)
    source_bits = dtype_bit_width(source_dtype)
    if destination_bits is None or source_bits is None or destination_bits != source_bits:
        return None
    total_bits = count * destination_bits
    if total_bits % 8:
        return None
    total_bytes = total_bits // 8
    return total_bytes if total_bytes in {4, 8, 16} else None


def valid_set_max_nreg(arguments: Any, scalar_evaluator: ScalarEvaluator) -> bool:
    if len(arguments) != 2:
        return False
    register_count = scalar_evaluator(arguments[0])
    action = scalar_evaluator(arguments[1])
    return (
        register_count is not None
        and 24 <= register_count <= 256
        and register_count % 8 == 0
        and action in {0, 1}
    )


def is_pure_payload_call(name: str, argument_count: int) -> bool:
    return (
        name == "tl.fast_rcp" and argument_count == 1
        or name in {"tir.reinterpret", "tirx.reinterpret"} and argument_count == 1
    )


def peel_static_byte_offsets(
    expression: Any,
    call_name: CallName,
    scalar_evaluator: ScalarEvaluator,
) -> tuple[Any, int] | None:
    byte_offset = 0
    while call_name(expression) in {
        "tirx.handle_add_byte_offset",
        "tir.handle_add_byte_offset",
    }:
        arguments = getattr(expression, "args", ())
        if len(arguments) != 2:
            return None
        offset = scalar_evaluator(arguments[1])
        if offset is None:
            return None
        byte_offset += offset
        expression = arguments[0]
    return expression, byte_offset


def _broadcast(values: tuple[int, ...], lanes: int) -> tuple[int, ...] | None:
    if len(values) == lanes:
        return values
    if len(values) == 1:
        return values * lanes
    return None


def _lane_arguments(*values: tuple[int, ...]) -> list[tuple[int, ...]] | None:
    lanes = max((len(value) for value in values), default=1)
    expanded = [_broadcast(value, lanes) for value in values]
    return None if any(value is None for value in expanded) else [value for value in expanded if value is not None]


def vector_values(expression: Any, scalar_evaluator: ScalarEvaluator) -> tuple[int, ...] | None:
    """Evaluate a finite integer TIR vector without executing arbitrary calls."""

    scalar = scalar_evaluator(expression)
    if scalar is not None:
        return (scalar,)
    name = type(expression).__name__
    if name == "Ramp":
        base = scalar_evaluator(expression.base)
        stride = scalar_evaluator(expression.stride)
        lanes = expression.lanes if isinstance(expression.lanes, int) else scalar_evaluator(expression.lanes)
        if base is None or stride is None or lanes is None or lanes <= 0:
            return None
        return tuple(base + lane * stride for lane in range(lanes))
    if name == "Broadcast":
        value = vector_values(expression.value, scalar_evaluator)
        lanes = expression.lanes if isinstance(expression.lanes, int) else scalar_evaluator(expression.lanes)
        if value is None or len(value) != 1 or lanes is None or lanes <= 0:
            return None
        return value * lanes
    if name == "Cast":
        return vector_values(expression.value, scalar_evaluator)
    if name == "Not":
        value = vector_values(expression.a, scalar_evaluator)
        return None if value is None else tuple(int(not item) for item in value)
    if name == "Select":
        condition = vector_values(expression.condition, scalar_evaluator)
        true_value = vector_values(expression.true_value, scalar_evaluator)
        false_value = vector_values(expression.false_value, scalar_evaluator)
        if condition is None or true_value is None or false_value is None:
            return None
        arguments = _lane_arguments(condition, true_value, false_value)
        if arguments is None:
            return None
        conditions, truths, falses = arguments
        return tuple(truth if condition else false for condition, truth, false in zip(conditions, truths, falses))
    if name == "Shuffle":
        flattened: list[int] = []
        for vector in expression.vectors:
            values = vector_values(vector, scalar_evaluator)
            if values is None:
                return None
            flattened.extend(values)
        indices: list[int] = []
        for index in expression.indices:
            values = vector_values(index, scalar_evaluator)
            if values is None:
                return None
            indices.extend(values)
        if any(index < 0 or index >= len(flattened) for index in indices):
            return None
        return tuple(flattened[index] for index in indices)

    def trunc_div(left: int, right: int) -> int:
        quotient = abs(left) // abs(right)
        return -quotient if (left < 0) != (right < 0) else quotient

    binary = {
        "Add": lambda a, b: a + b,
        "Sub": lambda a, b: a - b,
        "Mul": lambda a, b: a * b,
        "Div": trunc_div,
        "FloorDiv": lambda a, b: a // b,
        "Mod": lambda a, b: a - trunc_div(a, b) * b,
        "FloorMod": lambda a, b: a % b,
        "Min": min,
        "Max": max,
        "EQ": lambda a, b: int(a == b),
        "NE": lambda a, b: int(a != b),
        "LT": lambda a, b: int(a < b),
        "LE": lambda a, b: int(a <= b),
        "GT": lambda a, b: int(a > b),
        "GE": lambda a, b: int(a >= b),
        "And": lambda a, b: int(bool(a) and bool(b)),
        "Or": lambda a, b: int(bool(a) or bool(b)),
    }
    function = binary.get(name)
    left_node = getattr(expression, "a", None)
    right_node = getattr(expression, "b", None)
    if function is None and name == "Call":
        call = getattr(getattr(expression, "op", None), "name", "")
        function = {
            "tirx.bitwise_xor": lambda a, b: a ^ b,
            "tirx.bitwise_and": lambda a, b: a & b,
            "tirx.bitwise_or": lambda a, b: a | b,
            "tirx.shift_left": lambda a, b: a << b,
            "tirx.shift_right": lambda a, b: a >> b,
        }.get(call)
        arguments = getattr(expression, "args", ())
        if len(arguments) == 2:
            left_node, right_node = arguments
    if function is None or left_node is None or right_node is None:
        return None
    left = vector_values(left_node, scalar_evaluator)
    right = vector_values(right_node, scalar_evaluator)
    if left is None or right is None:
        return None
    arguments = _lane_arguments(left, right)
    if arguments is None:
        return None
    left_values, right_values = arguments
    if name in {"Div", "FloorDiv", "Mod", "FloorMod"} and any(value == 0 for value in right_values):
        return None
    if name == "Call" and getattr(getattr(expression, "op", None), "name", "") in {
        "tirx.shift_left",
        "tirx.shift_right",
    } and any(value < 0 for value in right_values):
        return None
    return tuple(function(left, right) for left, right in zip(left_values, right_values))
