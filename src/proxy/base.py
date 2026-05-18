"""Provider abstraction for LLM backends.

Concrete providers (Gemini, OpenAI, Anthropic) implement ``Provider``.
The Protocol is ``runtime_checkable`` so a registry/factory can validate
third-party implementations at boot time.
"""

from typing import Protocol, runtime_checkable

from src.errors import KVRaftError


class ProxyError(KVRaftError):
    """Base class for provider-layer failures."""


class ProviderTimeoutError(ProxyError):
    """Provider exceeded the configured timeout."""


class ProviderAPIError(ProxyError):
    """Provider's SDK returned an error or non-2xx response."""


class ProviderChainExhaustedError(ProxyError):
    """Every provider in the configured chain is open (failed or breaker-tripped)."""


@runtime_checkable
class Provider(Protocol):
    name: str

    async def complete(self, prompt: str) -> str: ...
