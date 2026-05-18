"""Anthropic provider using the official ``anthropic`` async client."""

from typing import cast

import anthropic
from anthropic import AsyncAnthropic

from src.config import get_settings
from src.proxy.base import ProviderAPIError, ProviderTimeoutError

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
DEFAULT_MAX_TOKENS = 1024


class AnthropicClient:
    name = "anthropic"

    def __init__(self, model: str = DEFAULT_MODEL, max_tokens: int = DEFAULT_MAX_TOKENS) -> None:
        api_key = get_settings().anthropic_api_key
        if not api_key:
            raise ProviderAPIError("anthropic_api_key is not configured")
        self._client = AsyncAnthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens

    async def complete(self, prompt: str) -> str:
        try:
            response = await self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
        except anthropic.APITimeoutError as exc:
            raise ProviderTimeoutError("Anthropic request timed out") from exc
        except anthropic.APIError as exc:
            raise ProviderAPIError(f"Anthropic request failed: {exc}") from exc
        if not response.content:
            return ""
        first = response.content[0]
        return cast(str, getattr(first, "text", ""))
