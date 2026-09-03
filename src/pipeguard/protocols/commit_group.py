from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CommittedGroup:
    sequence: int
    group_id: str
    commit_event: str
    issue_events: tuple[str, ...]
    completion_ids: dict[str, str]
    completed_endpoints: set[str] = field(default_factory=set)

    def completion_id(self, kind: str) -> str:
        return self.completion_ids[kind]

    def is_completed(self, kind: str) -> bool:
        return kind in self.completed_endpoints

    def mark_completed(self, kind: str) -> None:
        self.completed_endpoints.add(kind)


@dataclass(slots=True)
class CommitGroupState:
    protocol: str
    participants: tuple[str, ...]
    capacity: int
    completion_kinds: tuple[str, ...]
    staged: dict[str, list[str]] = field(default_factory=dict)
    next_group_seq: dict[str, int] = field(default_factory=dict)
    committed: dict[str, list[CommittedGroup]] = field(default_factory=dict)
    observed_frontiers: dict[str, dict[str, int]] = field(default_factory=dict)
    operand_fences: dict[str, set[str]] = field(default_factory=dict)

    def staged_operations(self, owner: str) -> list[str]:
        return self.staged.setdefault(owner, [])

    def groups(self, owner: str) -> list[CommittedGroup]:
        return self.committed.setdefault(owner, [])

    def pending(self, owner: str, completion_kind: str) -> list[CommittedGroup]:
        return [
            group for group in self.committed.get(owner, ())
            if not group.is_completed(completion_kind)
        ]

    def pending_groups(self, owner: str) -> list[CommittedGroup]:
        return [
            group for group in self.committed.get(owner, ())
            if any(
                not group.is_completed(kind)
                for kind in group.completion_ids
            )
        ]

    def required_for_wait(
        self,
        owner: str,
        wait_group_count: int,
        completion_kind: str,
    ) -> list[CommittedGroup]:
        groups = self.committed.get(owner, [])
        cutoff = max(0, len(groups) - wait_group_count)
        return [
            group for group in groups[:cutoff]
            if completion_kind in group.completion_ids
        ]

    def observed_frontier(self, owner: str, completion_kind: str) -> int:
        return self.observed_frontiers.get(owner, {}).get(completion_kind, -1)

    def set_observed_frontier(
        self,
        owner: str,
        completion_kind: str,
        sequence: int,
    ) -> None:
        self.observed_frontiers.setdefault(owner, {})[completion_kind] = sequence

    def snapshot(self, owner: str | None = None) -> dict[str, Any]:
        owners = [owner] if owner is not None else sorted(self.participants)
        return {
            "protocol": self.protocol,
            "capacity": self.capacity,
            "completion_kinds": list(self.completion_kinds),
            "owners": {
                item: {
                    "next_group_seq": self.next_group_seq.get(item, 0),
                    "staged_operations": list(self.staged.get(item, [])),
                    "operand_fences": sorted(self.operand_fences.get(item, set())),
                    "committed_groups": [
                        {
                            "sequence": group.sequence,
                            "group_id": group.group_id,
                            "commit_event": group.commit_event,
                            "issue_events": list(group.issue_events),
                            "completion_ids": dict(sorted(group.completion_ids.items())),
                            "completed_endpoints": sorted(group.completed_endpoints),
                        }
                        for group in self.committed.get(item, [])
                    ],
                    "observed_frontiers": dict(sorted(
                        self.observed_frontiers.get(item, {}).items()
                    )),
                }
                for item in owners
            },
        }
