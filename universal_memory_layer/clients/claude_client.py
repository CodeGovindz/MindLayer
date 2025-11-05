"""Claude client implementation using Anthropic API."""

import json
import time
from typing import List, Optional, Dict, Any
import requests

from .base import BaseLLMClient


class ClaudeClient(BaseLLMClient):
    """Claude client implementation using Anthropic API."""
    
    def __init__(self, api_key: str, model: str = "claude-3-sonnet-20240229"):
        """
        Initialize Claude client.
        
        Args:
            api_key: Anthropic API key
            model: Model name (default: claude-3-sonnet-20240229)
        """
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
        self.max_retries = 3
        self.retry_delay = 1.0
        self.anthropic_version = "2023-06-01"
    
    def chat(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Send message with context to Claude and return response.
        
        Args:
            message: The user message to send
            context: Optional list of context messages to prepend
            
        Returns:
            The Claude response as a string
            
        Raises:
            Exception: If the API call fails after retries
        """
        try:
            formatted_message = self._format_user_message(message, context)
            
            for attempt in range(self.max_retries):
                try:
                    response = self._make_api_request(formatted_message)
                    return self._extract_response(response)
                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    
        except Exception as e:
            return self.handle_error(e)
    
    def format_prompt(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Format message and context into Claude-specific prompt.
        
        Args:
            message: The user message
            context: Optional list of context messages
            
        Returns:
            Formatted prompt string for Claude
        """
        if context:
            context_text = "\n\n".join(context)
            return f"Here is some relevant context from our previous conversation:\n\n{context_text}\n\nHuman: {message}\n\nAssistant:"
        else:
            return f"Human: {message}\n\nAssistant:"
    
    def _format_user_message(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Format message and context for Claude API.
        
        Args:
            message: The user message
            context: Optional list of context messages
            
        Returns:
            Formatted message for Claude API
        """
        if context:
            context_text = "\n\n".join(context)
            return f"Here is some relevant context from our previous conversation:\n\n{context_text}\n\n{message}"
        else:
            return message
    
    def _make_api_request(self, message: str) -> Dict[str, Any]:
        """
        Make API request to Anthropic.
        
        Args:
            message: Formatted message string
            
        Returns:
            API response dictionary
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": self.anthropic_version,
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 1000,
            "messages": [
                {
                    "role": "user",
                    "content": message
                }
            ]
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 401:
            raise Exception("Invalid Anthropic API key. Please check your ANTHROPIC_API_KEY environment variable.")
        elif response.status_code == 429:
            raise Exception("Anthropic API rate limit exceeded. Please try again later.")
        elif response.status_code >= 400:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", {}).get("message", "")
                if not error_detail:
                    error_detail = error_data.get("message", "")
            except:
                error_detail = response.text
            raise Exception(f"Anthropic API error ({response.status_code}): {error_detail}")
        
        response.raise_for_status()
        return response.json()
    
    def _extract_response(self, response: Dict[str, Any]) -> str:
        """
        Extract the response text from Anthropic API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            The response text
            
        Raises:
            Exception: If response format is unexpected
        """
        try:
            content = response.get("content", [])
            if not content:
                raise Exception("No content returned from Anthropic API")
            
            # Claude API returns content as a list of content blocks
            text_content = ""
            for block in content:
                if block.get("type") == "text":
                    text_content += block.get("text", "")
            
            if not text_content:
                raise Exception("Empty response content from Anthropic API")
            
            return text_content.strip()
            
        except (KeyError, IndexError, TypeError) as e:
            raise Exception(f"Unexpected response format from Anthropic API: {e}")
    
    def handle_error(self, error: Exception) -> str:
        """
        Handle API errors gracefully with Claude-specific messages.
        
        Args:
            error: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_msg = str(error).lower()
        
        if "api key" in error_msg or "unauthorized" in error_msg:
            return "Claude API authentication failed. Please check your Anthropic API key."
        elif "rate limit" in error_msg:
            return "Claude API rate limit exceeded. Please wait a moment and try again."
        elif "timeout" in error_msg:
            return "Claude API request timed out. Please check your internet connection and try again."
        elif "connection" in error_msg:
            return "Unable to connect to Claude API. Please check your internet connection."
        else:
            return f"Claude API error: {str(error)}"