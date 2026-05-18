from unittest.mock import AsyncMock, MagicMock

import openai as openai_sdk
import pytest

from src.proxy import openai_client
from src.proxy.base import Provider, ProviderAPIError, ProviderTimeoutError


@pytest.fixture
def _configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock) -> MagicMock:
    ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(openai_client, "AsyncOpenAI", ctor)
    return ctor


def test_init_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(ProviderAPIError, match="not configured"):
        openai_client.OpenAIClient()


def test_openai_client_satisfies_provider_protocol(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    _install_mock_client(monkeypatch, MagicMock())
    assert isinstance(openai_client.OpenAIClient(), Provider)


async def test_complete_returns_response_text(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    message = MagicMock()
    message.content = "hi back"
    choice = MagicMock(message=message)
    response = MagicMock(choices=[choice])
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(return_value=response)
    _install_mock_client(monkeypatch, fake_client)

    result = await openai_client.OpenAIClient().complete("hi")
    assert result == "hi back"


async def test_complete_maps_timeout_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=openai_sdk.APITimeoutError(request=MagicMock())
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderTimeoutError):
        await openai_client.OpenAIClient().complete("hi")


async def test_complete_maps_api_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_client = MagicMock()
    fake_client.chat = MagicMock()
    fake_client.chat.completions = MagicMock()
    fake_client.chat.completions.create = AsyncMock(
        side_effect=openai_sdk.APIError("boom", request=MagicMock(), body=None)
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderAPIError):
        await openai_client.OpenAIClient().complete("hi")
