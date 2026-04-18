from unittest.mock import AsyncMock, MagicMock

import pytest
from google.api_core import exceptions as google_exceptions

from src.proxy import gemini_client
from src.proxy.base import Provider, ProviderAPIError, ProviderTimeoutError


@pytest.fixture
def _configured_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


def _install_mock_model(monkeypatch: pytest.MonkeyPatch, fake_model: MagicMock) -> MagicMock:
    configure = MagicMock()
    generative_model_ctor = MagicMock(return_value=fake_model)
    monkeypatch.setattr(gemini_client.genai, "configure", configure)
    monkeypatch.setattr(gemini_client.genai, "GenerativeModel", generative_model_ctor)
    return generative_model_ctor


def test_init_raises_when_api_key_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ProviderAPIError, match="not configured"):
        gemini_client.GeminiClient()


def test_init_configures_sdk_with_key(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_model = MagicMock()
    ctor = _install_mock_model(monkeypatch, fake_model)

    gemini_client.GeminiClient(model="gemini-test")

    ctor.assert_called_once_with("gemini-test")
    assert gemini_client.genai.configure.call_args.kwargs == {"api_key": "test-key"}  # type: ignore[attr-defined]


def test_gemini_client_satisfies_provider_protocol(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    _install_mock_model(monkeypatch, MagicMock())
    assert isinstance(gemini_client.GeminiClient(), Provider)


async def test_complete_returns_response_text(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_response = MagicMock(text="hello world")
    fake_model = MagicMock()
    fake_model.generate_content_async = AsyncMock(return_value=fake_response)
    _install_mock_model(monkeypatch, fake_model)

    client = gemini_client.GeminiClient()
    result = await client.complete("hi")

    assert result == "hello world"
    fake_model.generate_content_async.assert_awaited_once_with("hi")


async def test_complete_maps_deadline_exceeded_to_timeout(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_model = MagicMock()
    fake_model.generate_content_async = AsyncMock(
        side_effect=google_exceptions.DeadlineExceeded("slow")
    )
    _install_mock_model(monkeypatch, fake_model)

    client = gemini_client.GeminiClient()

    with pytest.raises(ProviderTimeoutError) as exc_info:
        await client.complete("hi")
    assert isinstance(exc_info.value.__cause__, google_exceptions.DeadlineExceeded)


async def test_complete_maps_generic_api_error(
    monkeypatch: pytest.MonkeyPatch, _configured_key: None
) -> None:
    fake_model = MagicMock()
    fake_model.generate_content_async = AsyncMock(
        side_effect=google_exceptions.GoogleAPIError("500")
    )
    _install_mock_model(monkeypatch, fake_model)

    client = gemini_client.GeminiClient()

    with pytest.raises(ProviderAPIError) as exc_info:
        await client.complete("hi")
    assert isinstance(exc_info.value.__cause__, google_exceptions.GoogleAPIError)
