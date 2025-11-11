# python
# 文件：`ace/llm_deepseek.py`
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
    m = _CODE_FENCE_RE.match(s)
    if m:
        inner = m.group("body").strip()
        return inner

    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = s[start : end + 1].strip()
        return candidate

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
        # 新增：默认采样参数（当调用方未显式传参时使用）
        default_temperature: Optional[float] = None,
        default_max_new_tokens: Optional[int] = None,
        default_top_p: Optional[float] = None,
    ) -> None:
        """Initialize DeepSeek client.

        Args:
            model: Model name (default: "deepseek-chat")
            api_key: DeepSeek API key (defaults to DEEPSEEK_API_KEY env var)
            base_url: Base URL for API (defaults to DEEPSEEK_BASE_URL env or https://api.deepseek.com)
            timeout: Request timeout in seconds
            max_retries: Maximum number of retries on failure
            retry_delay: Delay between retries in seconds
            default_temperature: Default temperature if not provided per call
            default_max_new_tokens: Default max new tokens if not provided per call
            default_top_p: Default top_p if not provided per call
        """
        super().__init__(model=model)

        self.api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError(
                "DeepSeek API key is required. "
                "Set DEEPSEEK_API_KEY environment variable or pass api_key parameter."
            )

        self.base_url = base_url or os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        # 默认采样参数（仅在调用未显式传参时生效）
        self.default_temperature = default_temperature
        self.default_max_new_tokens = default_max_new_tokens
        self.default_top_p = default_top_p

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
            **kwargs: temperature, max_new_tokens, top_p, max_tokens (显式传参优先)

        Returns:
            LLMResponse with text and raw response data
        """
        messages = [{"role": "user", "content": prompt}]

        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }

        # 显式传参优先；否则使用默认值；最后不设置则交由服务端默认
        temperature = kwargs.get("temperature", self.default_temperature)
        if temperature is not None:
            api_params["temperature"] = temperature

        # 兼容 max_tokens 与 max_new_tokens，两者以显式传入为先
        if "max_tokens" in kwargs:
            api_params["max_tokens"] = kwargs["max_tokens"]
        else:
            max_new = kwargs.get("max_new_tokens", self.default_max_new_tokens)
            if max_new is not None:
                api_params["max_tokens"] = max_new

        top_p = kwargs.get("top_p", self.default_top_p)
        if top_p is not None:
            api_params["top_p"] = top_p

        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries):
            try:
                response = self.client.post(
                    f"{self.base_url}/chat/completions",
                    json=api_params,
                )

                if response.status_code == 200:
                    data = response.json()
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
        s = (text or "").strip()
        if not s:
            return s
        s = _extract_json_block(s)
        return s

    def __del__(self):
        if hasattr(self, "client"):
            try:
                self.client.close()
            except Exception:
                pass