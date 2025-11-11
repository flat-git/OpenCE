"""DeepSeek LLM client using httpx for REST API calls (with JSON fence sanitization)."""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, Optional

import httpx

from .llm import LLMClient, LLMResponse


_CODE_FENCE_RE = re.compile(
    r"^\s*```(?:json)?\s*(?P<body>[\s\S]*?)\s*```\s*$", re.IGNORECASE
)


def _extract_json_block(text: str) -> str:
    """
    Try to extract a pure JSON object from possibly fenced or mixed text.
    Strategy:
      1) If ```json ... ``` fences exist, return inner.
      2) Else, locate the first '{' and the last '}' and return that slice.
      3) Fallback: stripped original.
    """
    s = text.strip()
    # Case 1: code fenced JSON
    m = _CODE_FENCE_RE.match(s)
    if m:
        inner = m.group("body").strip()
        return inner

    # Case 2: find the largest JSON object region
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1].strip()
        return candidate

    # Fallback: return stripped text
    return s


class DeepSeekClient(LLMClient):
    """LLM client for DeepSeek API using httpx for REST calls."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
    ) -> None:
        """Initialize DeepSeek client.

        Args:
            model: Model name (default: "deepseek-chat")
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: Base URL for API (defaults to DEEPSEEK_BASE_URL env or https://api.deepseek.com)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
        """
        super().__init__(model=model)

        # Get API key from parameter or environment
        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )

        # Get base URL from parameter or environment
        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # Create httpx client
        self.client = httpx.Client(
            timeout=timeout,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
        )

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate completion for the given prompt.

        Args:
            prompt: User prompt text
            **kwargs: Additional parameters like temperature, max_new_tokens, top_p

        Returns:
            LLMResponse with text and raw response data
        """
        messages = [{"role": "user", "content": prompt}]

        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        if "temperature" in kwargs:
            api_params["temperature"] = kwargs["temperature"]
        if "max_new_tokens" in kwargs:
            api_params["max_tokens"] = kwargs["max_new_tokens"]
        if "top_p" in kwargs:
            api_params["top_p"] = kwargs["top_p"]

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=api_params,
                )

                if response.status_code == 200:
                    data = response.json()
                    # Extract assistant message content
                    if "choices" in data and len(data["choices"]) > 0:
                        content = data["choices"][0].get("message", {}).get("content", "") or ""
                        sanitized = self._sanitize_text(content)
                        return LLMResponse(text=sanitized, raw=data)
                    else:
                        raise ValueError(f"Unexpected response format: {data}")
                else:
                    error_msg = f"API request failed with status {response.status_code}: {response.text}"
                    if attempt + 1 >= self.max_retries:
                        raise RuntimeError(error_msg)
                    last_error = RuntimeError(error_msg)
                    time.sleep(self.retry_delay * (attempt + 1))

            except httpx.HTTPError as exc:
                error_msg = f"HTTP error occurred: {exc}"
                if attempt + 1 >= self.max_retries:
                    raise RuntimeError(error_msg) from exc
                last_error = exc
                time.sleep(self.retry_delay * (attempt + 1))
            except Exception as exc:
                error_msg = f"Unexpected error: {exc}"
                if attempt + 1 >= self.max_retries:
                    raise RuntimeError(error_msg) from exc
                last_error = exc
                time.sleep(self.retry_delay * (attempt + 1))

        raise RuntimeError("DeepSeek API call failed after all retries") from last_error

    def _sanitize_text(self, text: str) -> str:
        """
        Sanitize response text:
          - strip whitespace
          - remove ```json ... ``` fences
          - extract the largest JSON object if mixed text appears
        """
        s = (text or "").strip()
        if not s:
            return s
        s = _extract_json_block(s)
        return s

    def __del__(self):
        """Clean up httpx client on deletion."""
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:
                pass