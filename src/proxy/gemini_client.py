"""Gemini provider using ``google.generativeai``.

The SDK is deprecated in favor of ``google.genai``; migration is deferred
until after the Day-3 demo. See DECISIONS.md.
"""

from typing import cast

import google.generativeai as genai
from google.api_core import exceptions as google_exceptions

from src.config import get_settings
from src.proxy.base import ProviderAPIError, ProviderTimeoutError

DEFAULT_MODEL = "gemini-2.5-flash-lite"


class GeminiClient:
    name = "gemini"

    def __init__(self, model: str = DEFAULT_MODEL) -> None:
        api_key = get_settings().gemini_api_key
        if not api_key:
            raise ProviderAPIError("gemini_api_key is not configured")
        genai.configure(api_key=api_key)
        self._model = genai.GenerativeModel(model)

    async def complete(self, prompt: str) -> str:
        try:
            response = await self._model.generate_content_async(prompt)
        except google_exceptions.DeadlineExceeded as exc:
            raise ProviderTimeoutError("Gemini request timed out") from exc
        except google_exceptions.GoogleAPIError as exc:
            raise ProviderAPIError(f"Gemini request failed: {exc}") from exc
        return cast(str, response.text)
