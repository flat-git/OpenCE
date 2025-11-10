"""DeepSeek LLM client using httpx for REST API calls."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import httpx

from .llm import LLMClient, LLMResponse


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
            }
        )

    def complete(self, prompt: str, **kwargs: Any) -> LLMResponse:
        """Generate completion for the given prompt.

        Args:
            prompt: User prompt text
            **kwargs: Additional parameters like temperature, max_new_tokens, top_p

        Returns:
            LLMResponse with text and raw response data
        """
        # Build chat messages
        messages = [{"role": "user", "content": prompt}]
        
        # Map kwargs to API parameters
        api_params: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
        }
        
        # Map common parameter names
        if "temperature" in kwargs:
            api_params["temperature"] = kwargs["temperature"]
        if "max_new_tokens" in kwargs:
            api_params["max_tokens"] = kwargs["max_new_tokens"]
        if "top_p" in kwargs:
            api_params["top_p"] = kwargs["top_p"]
        
        # Try with retries
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
                        content = data["choices"][0].get("message", {}).get("content", "")
                        return LLMResponse(text=self._sanitize_text(content), raw=data)
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
        """Sanitize response text by removing extra whitespace."""
        return text.strip()

    def __del__(self):
        """Clean up httpx client on deletion."""
        if hasattr(self, 'client'):
            self.client.close()
