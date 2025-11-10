"""Tests for DeepSeekClient."""

import json
import os
import unittest
from unittest.mock import MagicMock, patch

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ace import DeepSeekClient
from ace.llm import LLMResponse


class DeepSeekClientTest(unittest.TestCase):
    """Test DeepSeekClient functionality."""
    
    def test_initialization_with_api_key(self):
        """Test that client can be initialized with API key."""
        client = DeepSeekClient(api_key="test-key-123")
        self.assertEqual(client.api_key, "test-key-123")
        self.assertEqual(client.model, "deepseek-chat")
        self.assertEqual(client.base_url, "https://api.deepseek.com")
    
    def test_initialization_with_env_var(self):
        """Test that client reads API key from environment."""
        with patch.dict(os.environ, {"DEEPSEEK_API_KEY": "env-key-456"}):
            client = DeepSeekClient()
            self.assertEqual(client.api_key, "env-key-456")
    
    def test_initialization_without_api_key_fails(self):
        """Test that initialization fails without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as context:
                DeepSeekClient()
            self.assertIn("API key is required", str(context.exception))
    
    def test_custom_base_url(self):
        """Test custom base URL."""
        client = DeepSeekClient(
            api_key="test-key",
            base_url="https://custom.api.com"
        )
        self.assertEqual(client.base_url, "https://custom.api.com")
    
    def test_base_url_from_env(self):
        """Test base URL from environment variable."""
        with patch.dict(os.environ, {
            "DEEPSEEK_API_KEY": "test-key",
            "DEEPSEEK_BASE_URL": "https://env.api.com"
        }):
            client = DeepSeekClient()
            self.assertEqual(client.base_url, "https://env.api.com")
    
    @patch('httpx.Client.post')
    def test_complete_success(self, mock_post):
        """Test successful completion."""
        # Mock successful API response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [
                {
                    "message": {
                        "content": "Test response content"
                    }
                }
            ]
        }
        mock_post.return_value = mock_response
        
        client = DeepSeekClient(api_key="test-key")
        response = client.complete("Test prompt")
        
        self.assertIsInstance(response, LLMResponse)
        self.assertEqual(response.text, "Test response content")
        self.assertIsNotNone(response.raw)
        
        # Verify API call
        mock_post.assert_called_once()
        call_args = mock_post.call_args
        self.assertIn("json", call_args.kwargs)
        json_payload = call_args.kwargs["json"]
        self.assertEqual(json_payload["model"], "deepseek-chat")
        self.assertEqual(json_payload["messages"][0]["content"], "Test prompt")
    
    @patch('httpx.Client.post')
    def test_complete_with_kwargs(self, mock_post):
        """Test completion with additional kwargs."""
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Response"}}]
        }
        mock_post.return_value = mock_response
        
        client = DeepSeekClient(api_key="test-key")
        client.complete(
            "Test prompt",
            temperature=0.7,
            max_new_tokens=512,
            top_p=0.9
        )
        
        # Verify parameters are mapped correctly
        call_args = mock_post.call_args
        json_payload = call_args.kwargs["json"]
        self.assertEqual(json_payload.get("temperature"), 0.7)
        self.assertEqual(json_payload.get("max_tokens"), 512)
        self.assertEqual(json_payload.get("top_p"), 0.9)
    
    @patch('httpx.Client.post')
    def test_complete_retry_on_failure(self, mock_post):
        """Test retry logic on API failure."""
        # First two calls fail, third succeeds
        mock_response_fail = MagicMock()
        mock_response_fail.status_code = 500
        mock_response_fail.text = "Internal server error"
        
        mock_response_success = MagicMock()
        mock_response_success.status_code = 200
        mock_response_success.json.return_value = {
            "choices": [{"message": {"content": "Success after retry"}}]
        }
        
        mock_post.side_effect = [
            mock_response_fail,
            mock_response_fail,
            mock_response_success
        ]
        
        client = DeepSeekClient(api_key="test-key", retry_delay=0.1)
        response = client.complete("Test prompt")
        
        self.assertEqual(response.text, "Success after retry")
        self.assertEqual(mock_post.call_count, 3)
    
    @patch('httpx.Client.post')
    def test_complete_max_retries_exceeded(self, mock_post):
        """Test that exception is raised after max retries."""
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.text = "Server error"
        mock_post.return_value = mock_response
        
        client = DeepSeekClient(api_key="test-key", retry_delay=0.1)
        
        with self.assertRaises(RuntimeError) as context:
            client.complete("Test prompt")
        
        self.assertIn("failed with status 500", str(context.exception))
        self.assertEqual(mock_post.call_count, 3)  # max_retries default is 3


if __name__ == "__main__":
    unittest.main()
