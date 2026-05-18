from unittest.mock import AsyncMock, MagicMock

import anthropic
import pytest

from src.proxy import anthropic_client
from src.proxy.base import Provider, ProviderAPIError, ProviderTimeoutError


@pytest.fixture
def _configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")


def _install_mock_client(monkeypatch: pytest.MonkeyPatch, fake_client: MagicMock) -> MagicMock:
    ctor = MagicMock(return_value=fake_client)
    monkeypatch.setattr(anthropic_client, "AsyncAnthropic", ctor)
    return ctor


def test_init_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderAPIError, match="not configured"):
        anthropic_client.AnthropicClient()


def test_anthropic_client_satisfies_provider_protocol(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    _install_mock_client(monkeypatch, MagicMock())
    assert isinstance(anthropic_client.AnthropicClient(), Provider)


async def test_complete_returns_response_text(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    block = MagicMock()
    block.text = "hi from claude"
    response = MagicMock(content=[block])
    fake_client = MagicMock()
    fake_client.messages = MagicMock()
    fake_client.messages.create = AsyncMock(return_value=response)
    _install_mock_client(monkeypatch, fake_client)

    result = await anthropic_client.AnthropicClient().complete("hi")
    assert result == "hi from claude"


async def test_complete_maps_timeout_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_client = MagicMock()
    fake_client.messages = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=anthropic.APITimeoutError(request=MagicMock())
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderTimeoutError):
        await anthropic_client.AnthropicClient().complete("hi")


async def test_complete_maps_api_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_client = MagicMock()
    fake_client.messages = MagicMock()
    fake_client.messages.create = AsyncMock(
        side_effect=anthropic.APIError("boom", request=MagicMock(), body=None)
    )
    _install_mock_client(monkeypatch, fake_client)

    with pytest.raises(ProviderAPIError):
        await anthropic_client.AnthropicClient().complete("hi")
