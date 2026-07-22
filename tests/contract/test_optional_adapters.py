from types import SimpleNamespace

import pytest

from evalframe.models import ModelRequest
from evalframe.providers.anthropic import AnthropicProvider
from evalframe.providers.gemini import GeminiProvider
from evalframe.providers.openai import OpenAIProvider


def request() -> ModelRequest:
    return ModelRequest(
        case_id="case",
        system_prompt="system",
        user_input="input",
        output_json_schema={
            "type": "object",
            "properties": {"label": {"type": "string"}},
            "required": ["label"],
            "additionalProperties": False,
        },
        parameters={"model": "test-model", "temperature": 0},
    )


class AsyncMethod:
    def __init__(self, result: object) -> None:
        self.result = result
        self.kwargs: dict[str, object] = {}

    async def __call__(self, **kwargs: object) -> object:
        self.kwargs = kwargs
        return self.result


@pytest.mark.asyncio
async def test_openai_normalizes_strict_response() -> None:
    create = AsyncMethod(
        SimpleNamespace(
            output_text='{"label":"yes"}',
            usage=SimpleNamespace(input_tokens=4, output_tokens=2),
            id="response-id",
        )
    )
    client = SimpleNamespace(responses=SimpleNamespace(create=create))
    provider = OpenAIProvider(client=client)
    response = await provider.generate(request())
    assert response.parsed_output == {"label": "yes"}
    assert (response.input_tokens, response.output_tokens) == (4, 2)
    assert create.kwargs["text"] == {
        "format": {
            "type": "json_schema",
            "name": "output",
            "strict": True,
            "schema": request().output_json_schema,
        }
    }


@pytest.mark.asyncio
async def test_anthropic_normalizes_required_tool_response() -> None:
    create = AsyncMethod(
        SimpleNamespace(
            content=[SimpleNamespace(type="tool_use", input={"label": "yes"})],
            usage=SimpleNamespace(input_tokens=5, output_tokens=3),
            id="message-id",
            stop_reason="tool_use",
        )
    )
    client = SimpleNamespace(messages=SimpleNamespace(create=create))
    response = await AnthropicProvider(client=client).generate(request())
    assert response.parsed_output == {"label": "yes"}
    assert (response.input_tokens, response.output_tokens) == (5, 3)
    assert create.kwargs["tool_choice"] == {"type": "tool", "name": "submit_output"}


@pytest.mark.asyncio
async def test_gemini_normalizes_json_schema_response() -> None:
    generate = AsyncMethod(
        SimpleNamespace(
            text='{"label":"yes"}',
            usage_metadata=SimpleNamespace(prompt_token_count=6, candidates_token_count=4),
            response_id="gemini-id",
        )
    )
    client = SimpleNamespace(models=SimpleNamespace(generate_content=generate))
    response = await GeminiProvider(client=client).generate(request())
    assert response.parsed_output == {"label": "yes"}
    assert (response.input_tokens, response.output_tokens) == (6, 4)
    config = generate.kwargs["config"]
    assert isinstance(config, dict)
    assert config["response_json_schema"] == request().output_json_schema
