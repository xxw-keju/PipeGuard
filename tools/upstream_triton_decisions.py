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
    additional_evidence: tuple[Any, ...] = ()
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

    @classmethod
    def accepted_many(
        cls, mechanism: str, evidence: list[Any]
    ) -> "TritonEvidenceDecision":
        assert evidence
        return cls(
            mechanism,
            "ACCEPTED",
            "ACCEPTED",
            evidence=evidence[0],
            additional_evidence=tuple(evidence[1:]),
            candidate_id=evidence[0].candidate.candidate_id,
        )
