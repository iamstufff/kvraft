"""OpenAI provider using the official ``openai`` async client."""

import openai
from openai import AsyncOpenAI

from src.config import get_settings
from src.proxy.base import ProviderAPIError, ProviderTimeoutError

DEFAULT_MODEL = "gpt-5.4-mini"


class OpenAIClient:
    name = "openai"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = get_settings().openai_api_key
        if not api_key:
            raise ProviderAPIError("openai_api_key is not configured")
        self._client = AsyncOpenAI(api_key=api_key)
        self._model = model

    async def complete(self, prompt: str) -> str:
        try:
            response = await self._client.chat.completions.create(
                model=self._model,
                messages=[{"role": "user", "content": prompt}],
            )
        except openai.APITimeoutError as exc:
            raise ProviderTimeoutError("OpenAI request timed out") from exc
        except openai.APIError as exc:
            raise ProviderAPIError(f"OpenAI request failed: {exc}") from exc
        content = response.choices[0].message.content
        return content or ""
