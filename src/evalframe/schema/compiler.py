"""Compile a canonical schema without weakening its meaning."""

from evalframe.models import CompatibilityReport, ProviderCapabilities
from evalframe.schema.capabilities import check_compatibility


def compile_schema(
    schema: dict[str, object], capabilities: ProviderCapabilities
) -> tuple[dict[str, object], CompatibilityReport]:
    report = check_compatibility(schema, capabilities)
    return schema, report
