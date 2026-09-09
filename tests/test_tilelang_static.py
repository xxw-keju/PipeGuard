from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).parents[1]
sys.path.insert(0, str(ROOT / "tools"))

from tilelang_static import (  # noqa: E402
    cp_async_transfer_bytes,
    dtype_bit_width,
    dtype_byte_width,
    is_pure_payload_call,
    peel_static_byte_offsets,
    valid_set_max_nreg,
    vector_values,
)


@dataclass(frozen=True)
class Scalar:
    value: int


@dataclass(frozen=True)
class Cast:
    value: object


@dataclass(frozen=True)
class Ramp:
    base: object
    stride: object
    lanes: int


@dataclass(frozen=True)
class Broadcast:
    value: object
    lanes: int


@dataclass(frozen=True)
class Select:
    condition: object
    true_value: object
    false_value: object


@dataclass(frozen=True)
class Shuffle:
    vectors: list[object]
    indices: list[object]


@dataclass(frozen=True)
class Add:
    a: object
    b: object


@dataclass(frozen=True)
class Mul:
    a: object
    b: object


@dataclass(frozen=True)
class LT:
    a: object
    b: object


@dataclass(frozen=True)
class FakeDType:
    bits: int
    lanes: int = 1


@dataclass(frozen=True)
class Op:
    name: str


@dataclass(frozen=True)
class Call:
    op: Op
    args: tuple[object, ...]


def _scalar(expression: object) -> int | None:
    return expression.value if isinstance(expression, Scalar) else None


def test_vector_values_evaluates_real_shuffle_select_shape():
    thread = Scalar(0)
    selected = Shuffle(
        [
            Select(LT(Cast(thread), Scalar(2)), Add(Cast(thread), Scalar(1022)), Add(Cast(thread), Scalar(-2))),
            Select(LT(Cast(thread), Scalar(1)), Add(Cast(thread), Scalar(1023)), Add(Cast(thread), Scalar(-1))),
            Cast(thread),
            Add(Cast(thread), Scalar(1)),
        ],
        [Scalar(0), Scalar(1), Scalar(2), Scalar(3)],
    )
    expression = Add(Broadcast(Scalar(0), 4), Mul(selected, Broadcast(Scalar(1), 4)))
    assert vector_values(expression, _scalar) == (1022, 1023, 0, 1)


def test_vector_values_supports_ramp_and_rejects_incompatible_or_invalid_lanes():
    assert vector_values(Ramp(Scalar(3), Scalar(2), 4), _scalar) == (3, 5, 7, 9)
    assert vector_values(Add(Ramp(Scalar(0), Scalar(1), 2), Ramp(Scalar(0), Scalar(1), 3)), _scalar) is None
    assert vector_values(Shuffle([Ramp(Scalar(0), Scalar(1), 2)], [Scalar(2)]), _scalar) is None


def test_dtype_and_cp_async_width_include_fixed_vector_lanes():
    assert dtype_bit_width(FakeDType(16, 4)) == 64
    assert dtype_byte_width(FakeDType(16, 4)) == 8
    assert cp_async_transfer_bytes(2, FakeDType(16), FakeDType(16)) == 4
    assert cp_async_transfer_bytes(2, FakeDType(16, 4), FakeDType(16, 4)) == 16
    assert cp_async_transfer_bytes(1, FakeDType(16), FakeDType(32)) is None
    assert cp_async_transfer_bytes(1, FakeDType(4), FakeDType(4)) is None
    assert cp_async_transfer_bytes(3, FakeDType(16), FakeDType(16)) is None


def test_static_byte_offset_peeling_accumulates_only_resolved_wrappers():
    base = Scalar(7)
    wrapped = Call(
        Op("tirx.handle_add_byte_offset"),
        (Call(Op("tir.handle_add_byte_offset"), (base, Scalar(8))), Scalar(16)),
    )
    call_name = lambda node: node.op.name if isinstance(node, Call) else None
    assert peel_static_byte_offsets(wrapped, call_name, _scalar) == (base, 24)
    dynamic = Call(Op("tirx.handle_add_byte_offset"), (base, object()))
    assert peel_static_byte_offsets(dynamic, call_name, _scalar) is None


def test_exact_tilelang_effect_classification_is_narrow():
    assert valid_set_max_nreg((Scalar(40), Scalar(0)), _scalar)
    assert valid_set_max_nreg((Scalar(224), Scalar(1)), _scalar)
    assert not valid_set_max_nreg((Scalar(20), Scalar(0)), _scalar)
    assert not valid_set_max_nreg((Scalar(40), Scalar(2)), _scalar)
    assert not valid_set_max_nreg((Scalar(40),), _scalar)
    assert is_pure_payload_call("tl.fast_rcp", 1)
    assert is_pure_payload_call("tirx.reinterpret", 1)
    assert is_pure_payload_call("tir.reinterpret", 1)
    assert not is_pure_payload_call("tl.fast_rcp", 2)
    assert not is_pure_payload_call("tirx.reinterpret", 2)
    assert not is_pure_payload_call("tirx.call_extern", 1)
