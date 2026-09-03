from __future__ import annotations

from typing import Any


class FrontendError(Exception):
    def __init__(self, code: str, message: str, details: dict[str, Any] | None = None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.details = details or {}

    def diagnostic(self) -> dict[str, Any]:
        result: dict[str, Any] = {"code": self.code, "message": self.message}
        if self.details:
            result["details"] = self.details
        return result


def require_exact_keys(
    value: dict[str, Any],
    required: set[str],
    optional: set[str],
    location: str,
) -> None:
    if not isinstance(value, dict):
        raise FrontendError(
            "FRONTEND_FIELD_TYPE",
            "frontend snapshot value must be an object",
            {"location": location},
        )
    missing = sorted(required - value.keys())
    unknown = sorted(value.keys() - required - optional)
    if missing:
        raise FrontendError(
            "FRONTEND_FIELDS_MISSING",
            "frontend snapshot object is missing required fields",
            {"location": location, "fields": missing},
        )
    if unknown:
        raise FrontendError(
            "FRONTEND_FIELDS_UNKNOWN",
            "frontend snapshot object contains unknown fields",
            {"location": location, "fields": unknown},
        )
