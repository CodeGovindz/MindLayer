"""Groq client implementation using Groq API."""

import json
import time
from typing import List, Optional, Dict, Any
import requests

from .base import BaseLLMClient


class GroqClient(BaseLLMClient):
    """Groq client implementation using Groq API."""
    
    def __init__(self, api_key: str, model: str = "llama3-8b-8192"):
        """
        Initialize Groq client.
        
        Args:
            api_key: Groq API key
            model: Model name (default: llama3-8b-8192)
        """
        super().__init__(api_key)
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1/chat/completions"
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def chat(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Send message with context to Groq and return response.
        
        Args:
            message: The user message to send
            context: Optional list of context messages to prepend
            
        Returns:
            The Groq response as a string
            
        Raises:
            Exception: If the API call fails after retries
        """
        try:
            messages = self._format_messages(message, context)
            
            for attempt in range(self.max_retries):
                try:
                    response = self._make_api_request(messages)
                    return self._extract_response(response)
                except requests.exceptions.RequestException as e:
                    if attempt == self.max_retries - 1:
                        raise
                    time.sleep(self.retry_delay * (2 ** attempt))  # Exponential backoff
                    
        except Exception as e:
            return self.handle_error(e)
    
    def format_prompt(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Format message and context into Groq-specific prompt.
        
        Args:
            message: The user message
            context: Optional list of context messages
            
        Returns:
            Formatted prompt string for Groq
        """
        messages = self._format_messages(message, context)
        
        # Convert to a readable format for debugging/logging
        formatted_parts = []
        for msg in messages:
            role = msg["role"].upper()
            content = msg["content"]
            formatted_parts.append(f"{role}: {content}")
        
        return "\n\n".join(formatted_parts)
    
    def _format_messages(self, message: str, context: Optional[List[str]] = None) -> List[Dict[str, str]]:
        """
        Format message and context into Groq messages format.
        
        Args:
            message: The user message
            context: Optional list of context messages
            
        Returns:
            List of message dictionaries in Groq format
        """
        messages = []
        
        # Add system message if context is provided
        if context:
            context_text = "\n\n".join(context)
            messages.append({
                "role": "system",
                "content": f"Here is some relevant context from our previous conversation:\n\n{context_text}\n\nPlease use this context to inform your response to the following message."
            })
        
        # Add the user message
        messages.append({
            "role": "user",
            "content": message
        })
        
        return messages
    
    def _make_api_request(self, messages: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Make API request to Groq.
        
        Args:
            messages: List of message dictionaries
            
        Returns:
            API response dictionary
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": 0.7,
            "max_tokens": 1000,
            "top_p": 1,
            "stream": False
        }
        
        response = requests.post(
            self.base_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 401:
            raise Exception("Invalid Groq API key. Please check your GROQ_API_KEY environment variable.")
        elif response.status_code == 429:
            raise Exception("Groq API rate limit exceeded. Please try again later.")
        elif response.status_code >= 400:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except:
                error_detail = response.text
            raise Exception(f"Groq API error ({response.status_code}): {error_detail}")
        
        response.raise_for_status()
        return response.json()
    
    def _extract_response(self, response: Dict[str, Any]) -> str:
        """
        Extract the response text from Groq API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            The response text
            
        Raises:
            Exception: If response format is unexpected
        """
        try:
            choices = response.get("choices", [])
            if not choices:
                raise Exception("No response choices returned from Groq API")
            
            message = choices[0].get("message", {})
            content = message.get("content", "")
            
            if not content:
                raise Exception("Empty response content from Groq API")
            
            return content.strip()
            
        except (KeyError, IndexError, TypeError) as e:
            raise Exception(f"Unexpected response format from Groq API: {e}")
    
    def handle_error(self, error: Exception) -> str:
        """
        Handle API errors gracefully with Groq-specific messages.
        
        Args:
            error: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_msg = str(error).lower()
        
        if "api key" in error_msg or "unauthorized" in error_msg:
            return "Groq API authentication failed. Please check your Groq API key."
        elif "rate limit" in error_msg:
            return "Groq API rate limit exceeded. Please wait a moment and try again."
        elif "timeout" in error_msg:
            return "Groq API request timed out. Please check your internet connection and try again."
        elif "connection" in error_msg:
            return "Unable to connect to Groq API. Please check your internet connection."
        else:
            return f"Groq API error: {str(error)}"