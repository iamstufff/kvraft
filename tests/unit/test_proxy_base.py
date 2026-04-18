import pytest

from src.errors import KVRaftError
from src.proxy.base import (
    Provider,
    ProviderAPIError,
    ProviderTimeoutError,
    ProxyError,
)


def test_proxy_errors_inherit_from_kvraft_base() -> None:
    assert issubclass(ProxyError, KVRaftError)
    assert issubclass(ProviderTimeoutError, ProxyError)
    assert issubclass(ProviderAPIError, ProxyError)


class _FakeProvider:
    name = "fake"

    async def complete(self, prompt: str) -> str:
        return f"response for: {prompt}"


class _IncompleteProvider:
    name = "incomplete"


def test_provider_protocol_accepts_matching_shape() -> None:
    assert isinstance(_FakeProvider(), Provider)


def test_provider_protocol_rejects_missing_method() -> None:
    assert not isinstance(_IncompleteProvider(), Provider)


async def test_fake_provider_returns_prompt_derived_response() -> None:
    provider: Provider = _FakeProvider()
    assert await provider.complete("hi") == "response for: hi"


def test_proxy_error_wraps_context_via_chaining() -> None:
    source = RuntimeError("sdk blew up")
    try:
        raise ProviderAPIError("wrap") from source
    except ProviderAPIError as exc:
        assert exc.__cause__ is source
    else:
        pytest.fail("ProviderAPIError was not raised")
