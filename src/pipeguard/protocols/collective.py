from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CollectiveBarrierState:
    participants: tuple[str, ...]
    capacity: int
    invocation: int = 0
    arrivals: dict[str, tuple[str, int]] = field(default_factory=dict)
    publications: set[int] = field(default_factory=set)
    observations: dict[int, set[str]] = field(default_factory=dict)

    @property
    def arrived_count(self) -> int:
        return sum(count for _event, count in self.arrivals.values())

    def snapshot(self) -> dict[str, Any]:
        return {
            "invocation": self.invocation,
            "capacity": self.capacity,
            "arrived_count": self.arrived_count,
            "arrived_actors": sorted(self.arrivals),
            "missing_actors": sorted(set(self.participants) - self.arrivals.keys()),
            "publications": sorted(self.publications),
            "observations": {
                str(invocation): sorted(actors)
                for invocation, actors in sorted(self.observations.items())
            },
        }
