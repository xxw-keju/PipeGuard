from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .common import VisibilityMask


@dataclass(frozen=True, slots=True)
class VisibilityTransfer:
    event: str
    actor: str
    from_domain: str
    to_domain: str
    scope: str
    regions: tuple[str, ...]
    visibility_mask: VisibilityMask

    def snapshot(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "actor": self.actor,
            "from_domain": self.from_domain,
            "to_domain": self.to_domain,
            "scope": self.scope,
            "regions": list(self.regions),
            "visibility_mask": self.visibility_mask.artifact(),
        }
