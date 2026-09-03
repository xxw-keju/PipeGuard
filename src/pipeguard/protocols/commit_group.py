from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class CommittedGroup:
    sequence: int
    group_id: str
    commit_event: str
    issue_events: tuple[str, ...]
    completion_id: str
    completed: bool = False


@dataclass(slots=True)
class CommitGroupState:
    protocol: str
    participants: tuple[str, ...]
    capacity: int
    staged: dict[str, list[str]] = field(default_factory=dict)
    next_group_seq: dict[str, int] = field(default_factory=dict)
    committed: dict[str, list[CommittedGroup]] = field(default_factory=dict)
    observed_frontier: dict[str, int] = field(default_factory=dict)

    def staged_operations(self, owner: str) -> list[str]:
        return self.staged.setdefault(owner, [])

    def groups(self, owner: str) -> list[CommittedGroup]:
        return self.committed.setdefault(owner, [])

    def pending(self, owner: str) -> list[CommittedGroup]:
        return [group for group in self.committed.get(owner, ()) if not group.completed]

    def required_for_wait(self, owner: str, wait_group_count: int) -> list[CommittedGroup]:
        groups = self.committed.get(owner, [])
        cutoff = max(0, len(groups) - wait_group_count)
        return groups[:cutoff]

    def snapshot(self, owner: str | None = None) -> dict[str, Any]:
        owners = [owner] if owner is not None else sorted(self.participants)
        return {
            "protocol": self.protocol,
            "capacity": self.capacity,
            "owners": {
                item: {
                    "next_group_seq": self.next_group_seq.get(item, 0),
                    "staged_operations": list(self.staged.get(item, [])),
                    "committed_groups": [
                        {
                            "sequence": group.sequence,
                            "group_id": group.group_id,
                            "commit_event": group.commit_event,
                            "issue_events": list(group.issue_events),
                            "completion_id": group.completion_id,
                            "completed": group.completed,
                        }
                        for group in self.committed.get(item, [])
                    ],
                    "observed_frontier": self.observed_frontier.get(item, -1),
                }
                for item in owners
            },
        }
