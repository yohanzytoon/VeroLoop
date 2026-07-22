import pytest

from evalframe.models import ModelRequest
from evalframe.providers.fake import FakeProvider


@pytest.mark.asyncio
async def test_fake_provider_contract() -> None:
    provider = FakeProvider(lambda request: {"ok": request.case_id})
    assert provider.name == "fake"
    assert provider.capabilities().native_structured_output
    response = await provider.generate(
        ModelRequest(
            case_id="case", user_input="hello", output_json_schema={"type": "object"}, parameters={}
        )
    )
    assert response.parsed_output == {"ok": "case"}
    assert response.input_tokens is not None
    assert response.output_tokens is not None
    assert response.provider_request_id
    assert not hasattr(response, "sdk_response")


def test_optional_adapters_are_lazy() -> None:
    import evalframe.providers

    assert evalframe.providers.FakeProvider is FakeProvider
