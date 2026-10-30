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
    completion_kinds: tuple[str, ...]
    staged: dict[str, list[str]] = field(default_factory=dict)
    next_group_seq: dict[str, int] = field(default_factory=dict)
    committed: dict[str, list[CommittedGroup]] = field(default_factory=dict)
    observed_frontiers: dict[str, dict[str, int]] = field(default_factory=dict)
    operand_fences: dict[str, set[str]] = field(default_factory=dict)
    next_incomplete: dict[str, dict[str, int]] = field(default_factory=dict)
    known_completion_ids: set[str] = field(default_factory=set)
    mbarrier_tracked: set[str] = field(default_factory=set)

    def staged_operations(self, owner: str) -> list[str]:
        return self.staged.setdefault(owner, [])

    def groups(self, owner: str) -> list[CommittedGroup]:
        return self.committed.setdefault(owner, [])

    def append_group(self, owner: str, group: CommittedGroup) -> int:
        duplicates = self.known_completion_ids.intersection(group.completion_ids.values())
        if duplicates:
            raise ValueError(f"duplicate completion ids: {sorted(duplicates)}")
        groups = self.groups(owner)
        groups.append(group)
        self.known_completion_ids.update(group.completion_ids.values())
        frontiers = self.next_incomplete.setdefault(owner, {})
        advances = 0
        for completion_kind in group.completion_ids:
            frontier = frontiers.get(completion_kind, 0)
            while frontier < len(groups) and groups[frontier].is_completed(completion_kind):
                frontier += 1
                advances += 1
            frontiers[completion_kind] = frontier
        return advances

    def duplicate_completion_ids(self, completion_ids: dict[str, str]) -> list[str]:
        return sorted(self.known_completion_ids.intersection(completion_ids.values()))

    def next_pending_group(
        self,
        owner: str,
        completion_kind: str,
    ) -> CommittedGroup | None:
        groups = self.committed.get(owner, ())
        frontier = self.next_incomplete.get(owner, {}).get(completion_kind, 0)
        return groups[frontier] if frontier < len(groups) else None

    def has_pending(self, owner: str, completion_kind: str) -> bool:
        return self.next_pending_group(owner, completion_kind) is not None

    def mark_completed(
        self,
        owner: str,
        completion_kind: str,
        group: CommittedGroup | None = None,
    ) -> tuple[CommittedGroup, int]:
        if group is None:
            group = self.next_pending_group(owner, completion_kind)
        if group is None:
            raise ValueError("no pending completion endpoint")
        group.mark_completed(completion_kind)
        groups = self.committed[owner]
        frontier = self.next_incomplete[owner][completion_kind]
        advances = 0
        while frontier < len(groups) and groups[frontier].is_completed(completion_kind):
            frontier += 1
            advances += 1
        self.next_incomplete[owner][completion_kind] = frontier
        return group, advances

    def required_frontier_group(
        self,
        owner: str,
        wait_group_count: int,
        completion_kind: str,
    ) -> CommittedGroup | None:
        groups = self.committed.get(owner, ())
        cutoff = max(0, len(groups) - wait_group_count)
        if cutoff == 0:
            return None
        group = groups[cutoff - 1]
        return group if completion_kind in group.completion_ids else None

    def wait_is_satisfied(
        self,
        owner: str,
        wait_group_count: int,
        completion_kind: str,
    ) -> bool:
        cutoff = max(0, len(self.committed.get(owner, ())) - wait_group_count)
        return self.next_incomplete.get(owner, {}).get(completion_kind, 0) >= cutoff

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
