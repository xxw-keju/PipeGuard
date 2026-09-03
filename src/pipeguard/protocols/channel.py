from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True)
class ChannelSlotState:
    slot: int
    logical_phase: int = 0
    state: Literal[
        "PRODUCER_AVAILABLE",
        "PRODUCER_OWNED",
        "COMMITTED",
        "CONSUMER_AVAILABLE",
        "CONSUMER_OWNED",
    ] = "PRODUCER_AVAILABLE"
    producer: str | None = None
    consumer: str | None = None
    acquire_event: str | None = None
    commit_event: str | None = None
    handoff_id: str | None = None
    wait_event: str | None = None

    def snapshot(self) -> dict[str, Any]:
        return {
            "slot": self.slot,
            "logical_phase": self.logical_phase,
            "state": self.state,
            "producer": self.producer,
            "consumer": self.consumer,
            "acquire_event": self.acquire_event,
            "commit_event": self.commit_event,
            "handoff_id": self.handoff_id,
            "wait_event": self.wait_event,
        }


@dataclass(slots=True)
class ProducerConsumerChannelState:
    participants: tuple[str, ...]
    capacity: int
    initialized: bool = False
    create_event: str | None = None
    slots: dict[int, ChannelSlotState] = field(default_factory=dict)
    handoffs: dict[tuple[int, int], str] = field(default_factory=dict)
    releases: dict[tuple[int, int], str] = field(default_factory=dict)

    def initialize(self, event_id: str) -> None:
        self.initialized = True
        self.create_event = event_id
        self.slots = {slot: ChannelSlotState(slot) for slot in range(self.capacity)}

    def snapshot(self, slot: int | None = None) -> dict[str, Any]:
        selected = [slot] if slot is not None and slot in self.slots else sorted(self.slots)
        return {
            "initialized": self.initialized,
            "create_event": self.create_event,
            "capacity": self.capacity,
            "participants": list(self.participants),
            "slots": {str(item): self.slots[item].snapshot() for item in selected},
            "handoffs": [
                {"slot": item_slot, "logical_phase": phase, "transition": identifier}
                for (item_slot, phase), identifier in sorted(self.handoffs.items())
            ],
            "releases": [
                {"slot": item_slot, "logical_phase": phase, "event": event}
                for (item_slot, phase), event in sorted(self.releases.items())
            ],
        }
