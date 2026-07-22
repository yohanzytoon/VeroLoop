from enum import StrEnum
from typing import Literal

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from evalframe.models import ProviderCapabilities
from evalframe.schema import canonicalize, compile_schema, schema_for, validate_output


class Kind(StrEnum):
    A = "a"
    B = "b"


class Child(BaseModel):
    value: int


class Output(BaseModel):
    model_config = ConfigDict(extra="forbid")
    kind: Kind
    children: list[Child]
    optional: str | None = None
    choice: int | Literal["auto"]


def test_nested_schema_is_stable_and_validates() -> None:
    assert schema_for(Output) == schema_for(Output)
    assert list(canonicalize({"z": 1, "a": 2})) == ["a", "z"]
    value = validate_output(Output, {"kind": "a", "children": [{"value": 1}], "choice": "auto"})
    assert value["children"] == [{"value": 1}]


def test_extra_fields_and_invalid_enum_are_rejected() -> None:
    with pytest.raises(ValidationError):
        validate_output(Output, {"kind": "wrong", "children": [], "choice": 1, "extra": True})


def test_unsupported_keyword_is_a_hard_error() -> None:
    caps = ProviderCapabilities(
        native_structured_output=True,
        strict_tool_schema=False,
        json_mode=True,
        streaming=False,
        usage_reporting=False,
        supported_json_schema_keywords=frozenset({"type", "properties", "required"}),
    )
    _, report = compile_schema(schema_for(Output), caps)
    assert not report.supported
    assert "anyOf" in report.unsupported_keywords
    assert "children" not in report.unsupported_keywords
    assert report.transformed_keywords == []
