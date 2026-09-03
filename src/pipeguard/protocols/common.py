from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RESOURCE_KINDS = frozenset({
    "mbarrier",
    "cta_barrier",
    "named_barrier",
    "async_group",
    "channel",
})
PROTOCOL_KINDS = frozenset({"CP_ASYNC", "WGMMA", "TMA_STORE"})
COMPLETION_KINDS = frozenset({
    "GROUP_COMPLETE",
    "OPERANDS_RELEASED",
    "RESULT_READY",
    "SOURCE_RELEASED",
    "GLOBAL_COMPLETE",
})
PROXY_DOMAINS = frozenset({"GENERIC", "ASYNC_COPY", "TMA", "TENSOR_CORE"})
SCOPES = frozenset({"CTA"})
WAIT_MODES = frozenset({"READ_ONLY", "GLOBAL"})


@dataclass(frozen=True, slots=True)
class VisibilityMask:
    shared: bool
    global_read: bool
    global_write: bool
    tensor_read: bool
    tensor_write: bool
    control_only: bool

    @classmethod
    def from_mapping(cls, value: dict[str, Any]) -> "VisibilityMask":
        return cls(**value)

    def artifact(self) -> dict[str, bool]:
        return {
            "shared": self.shared,
            "global_read": self.global_read,
            "global_write": self.global_write,
            "tensor_read": self.tensor_read,
            "tensor_write": self.tensor_write,
            "control_only": self.control_only,
        }
