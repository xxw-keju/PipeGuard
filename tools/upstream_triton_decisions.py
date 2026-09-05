from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TritonEvidenceDecision:
    """One relevant Triton compile or launch evidence decision."""

    mechanism: str
    status: str
    code: str
    evidence: Any | None = None
    candidate_id: str | None = None

    @classmethod
    def not_applicable(cls, mechanism: str) -> "TritonEvidenceDecision":
        return cls(mechanism, "NOT_APPLICABLE", "NOT_APPLICABLE")

    @classmethod
    def rejected(
        cls,
        mechanism: str,
        code: str,
        *,
        candidate_id: str | None = None,
    ) -> "TritonEvidenceDecision":
        return cls(mechanism, "REJECTED", code, candidate_id=candidate_id)

    @classmethod
    def accepted(cls, mechanism: str, evidence: Any) -> "TritonEvidenceDecision":
        return cls(
            mechanism,
            "ACCEPTED",
            "ACCEPTED",
            evidence=evidence,
            candidate_id=evidence.candidate.candidate_id,
        )
