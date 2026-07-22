"""Safe, serializable exceptions."""

from __future__ import annotations

import re

from pydantic import BaseModel

_AUTHORIZATION = re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?\S+")
_SECRET = re.compile(r"(?i)(api[-_ ]?key|bearer|token|secret)([\s:=]+)([^\s,;]+)")


def sanitize_message(message: str) -> str:
    """Remove common credential forms from error text."""
    message = _AUTHORIZATION.sub(r"\1[REDACTED]", message)
    return _SECRET.sub(r"\1\2[REDACTED]", message)[:2000]


class EvalFrameException(Exception):
    """Base exception with a safe public message."""

    def __init__(self, message: str) -> None:
        super().__init__(sanitize_message(message))


class ConfigurationError(EvalFrameException):
    """Invalid evaluation configuration."""


class SchemaCompatibilityError(EvalFrameException):
    """A provider cannot preserve the canonical schema."""


class ProviderError(EvalFrameException):
    """Normalized provider failure."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


class EvaluationError(BaseModel):
    """Failure safe for persistence in an evaluation report."""

    kind: str
    message: str
    retryable: bool = False

    @classmethod
    def from_exception(cls, exc: Exception) -> EvaluationError:
        return cls(
            kind=type(exc).__name__,
            message=sanitize_message(str(exc)),
            retryable=isinstance(exc, ProviderError) and exc.retryable,
        )
