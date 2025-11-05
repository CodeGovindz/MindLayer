"""Gemini client implementation using Google AI API."""

import json
import time
from typing import List, Optional, Dict, Any
import requests

from .base import BaseLLMClient


class GeminiClient(BaseLLMClient):
    """Gemini client implementation using Google AI API."""
    
    def __init__(self, api_key: str, model: str = "gemini-pro"):
        """
        Initialize Gemini client.
        
        Args:
            api_key: Google AI API key
            model: Model name (default: gemini-pro)
        """
        super().__init__(api_key)
        self.model = model
        self.base_url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        self.max_retries = 3
        self.retry_delay = 1.0
    
    def chat(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Send message with context to Gemini and return response.
        
        Args:
            message: The user message to send
            context: Optional list of context messages to prepend
            
        Returns:
            The Gemini response as a string
            
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
        Format message and context into Gemini-specific prompt.
        
        Args:
            message: The user message
            context: Optional list of context messages
            
        Returns:
            Formatted prompt string for Gemini
        """
        if context:
            context_text = "\n\n".join(context)
            return f"Context from previous conversation:\n{context_text}\n\nUser: {message}"
        else:
            return f"User: {message}"
    
    def _format_user_message(self, message: str, context: Optional[List[str]] = None) -> str:
        """
        Format message and context for Gemini API.
        
        Args:
            message: The user message
            context: Optional list of context messages
            
        Returns:
            Formatted message for Gemini API
        """
        if context:
            context_text = "\n\n".join(context)
            return f"Context from previous conversation:\n{context_text}\n\n{message}"
        else:
            return message
    
    def _make_api_request(self, message: str) -> Dict[str, Any]:
        """
        Make API request to Google AI.
        
        Args:
            message: Formatted message string
            
        Returns:
            API response dictionary
            
        Raises:
            requests.exceptions.RequestException: If API request fails
        """
        url = f"{self.base_url}?key={self.api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": message
                        }
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1000,
                "topP": 0.8,
                "topK": 10
            }
        }
        
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 400:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except:
                error_detail = response.text
            
            if "api key" in error_detail.lower():
                raise Exception("Invalid Google AI API key. Please check your GOOGLE_API_KEY environment variable.")
            else:
                raise Exception(f"Google AI API error: {error_detail}")
        elif response.status_code == 429:
            raise Exception("Google AI API rate limit exceeded. Please try again later.")
        elif response.status_code >= 400:
            error_detail = ""
            try:
                error_data = response.json()
                error_detail = error_data.get("error", {}).get("message", "")
            except:
                error_detail = response.text
            raise Exception(f"Google AI API error ({response.status_code}): {error_detail}")
        
        response.raise_for_status()
        return response.json()
    
    def _extract_response(self, response: Dict[str, Any]) -> str:
        """
        Extract the response text from Google AI API response.
        
        Args:
            response: API response dictionary
            
        Returns:
            The response text
            
        Raises:
            Exception: If response format is unexpected
        """
        try:
            candidates = response.get("candidates", [])
            if not candidates:
                raise Exception("No candidates returned from Google AI API")
            
            content = candidates[0].get("content", {})
            parts = content.get("parts", [])
            
            if not parts:
                raise Exception("No content parts returned from Google AI API")
            
            text_content = ""
            for part in parts:
                text_content += part.get("text", "")
            
            if not text_content:
                raise Exception("Empty response content from Google AI API")
            
            return text_content.strip()
            
        except (KeyError, IndexError, TypeError) as e:
            raise Exception(f"Unexpected response format from Google AI API: {e}")
    
    def handle_error(self, error: Exception) -> str:
        """
        Handle API errors gracefully with Gemini-specific messages.
        
        Args:
            error: The exception that occurred
            
        Returns:
            User-friendly error message
        """
        error_msg = str(error).lower()
        
        if "api key" in error_msg or "unauthorized" in error_msg:
            return "Gemini API authentication failed. Please check your Google AI API key."
        elif "rate limit" in error_msg:
            return "Gemini API rate limit exceeded. Please wait a moment and try again."
        elif "timeout" in error_msg:
            return "Gemini API request timed out. Please check your internet connection and try again."
        elif "connection" in error_msg:
            return "Unable to connect to Gemini API. Please check your internet connection."
        else:
            return f"Gemini API error: {str(error)}"